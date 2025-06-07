import math
import multiprocessing
import os
import platform
import threading
import time
from multiprocessing import Lock
from multiprocessing.managers import BaseProxy
from multiprocessing.queues import Queue

import pulp
import streamlit as st
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx, get_script_run_ctx


def start_ui_status_updater(ui_status, status_queue, total=None):
    # Zentraler Speicher für aktuellen Status
    parallel_status_map = {}
    parallel_status_lock = Lock()
    completed_set = set()
    stop_event = threading.Event()

    def updater():
        stopped = False
        placeholder = None

        while not stopped:
            # Ensure last run is executed
            stopped = stop_event.is_set()

            while not status_queue.empty():
                try:
                    store_id, dept_id, label, state = status_queue.get_nowait()
                except Exception:
                    continue

                key = (store_id, dept_id)
                with parallel_status_lock:
                    if label is None or state != "running":  # Signal zur Entfernung (z.B. bei Abschluss)
                        parallel_status_map.pop(key, None)
                        completed_set.add(key)
                    else:
                        parallel_status_map[key] = label

            # Fortschrittsdaten auslesen
            with parallel_status_lock:
                current_done = len(completed_set)
                current_running = len(parallel_status_map)
                progress = f"{current_done} / {total}" if total else f"{current_done} abgeschlossen"

                # Überschrift für Statusbereich
                status_label = f"Optimierung wird durchgeführt... ({progress})"

                # Details aufbereiten
                details = []

                if current_running > 0:
                    details.append("**Aktive Optimierungen:**")
                    for (s, d), msg in sorted(parallel_status_map.items()):
                        details.append(f"- Store `{s}` / Dept `{d}`: _{msg}_")

                if current_done > 0:
                    details.append("**Abgeschlossen:**")
                    for (s, d) in sorted(completed_set):
                        if (s, d) not in parallel_status_map:
                            details.append(f"- Store `{s}` / Dept `{d}`")

            # UI-Update zentral ausführen
            placeholder = update_status(
                ui_status,
                label=status_label,
                state="running" if current_running > 0 else "complete",
                details=details,
                placeholder=placeholder
            )

            time.sleep(0.5)  # Regelmäßiges Polling

    thread = threading.Thread(target=updater, daemon=True)
    # Expose context to thread
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()
    return stop_event


def report_status(status_object, store_id, dept_id, label, state="running", expanded=None, details=None):
    if isinstance(status_object, Queue) or (
            isinstance(status_object, BaseProxy) and hasattr(status_object, "put") and hasattr(status_object, "get")):
        status_object.put((store_id, dept_id, label, state))
    elif isinstance(status_object, StatusContainer):
        update_status(status_object, label, state, expanded, details)


def update_status(ui_status, label, state="running", expanded=None, details=None, placeholder=None):
    if ui_status:
        # Initialer UI-Block für Statusnachrichten (wird jedes Mal neu geschrieben)
        with ui_status:
            # neuer Container für sauberes Leeren
            if placeholder is None:
                placeholder = st.empty()

            # Inhalte einfügen
            with placeholder.container():
                if details:
                    for line in details:
                        st.markdown(line)

        ui_status.update(label=label, state=state, expanded=expanded)
    return placeholder


# Automatische Pfadwahl je nach Betriebssystem
def get_default_cplex_path():
    system = platform.system()
    if system == "Windows":
        return r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe"
    elif system == "Linux":
        return "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex"
    elif system == "Darwin":  # macOS
        return "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex"
    else:
        return None


def create_solver(solver_timeout=150,
                  cplex_path=get_default_cplex_path(),
                  multithreading=True, debug=False):
    # Verwende mindestens 1, am besten 5% der CPU-Kerne, wenn wir mit anderen Solvern gleichzeitig ausführen
    thread_count = max(1, math.ceil(multiprocessing.cpu_count() * 0.05))
    if multithreading:
        # Benutze alle CPU-Kerne, wenn wir alleine ausführen
        thread_count = multiprocessing.cpu_count()

    # Versuche zuerst, CPLEX als Solver zu verwenden
    try:
        # Prüfen, ob die Datei existiert
        if os.path.isfile(cplex_path):
            solver = pulp.CPLEX_CMD(path=cplex_path, timeLimit=solver_timeout, msg=debug, threads=thread_count,
                                    logPath="cplex.log")
        else:
            # Wenn Pfad nicht gültig, versuche ohne Pfad (vorausgesetzt CPLEX ist im PATH)
            solver = pulp.CPLEX_CMD(timeLimit=solver_timeout, msg=debug, threads=thread_count, logPath="cplex.log")

        # Testweise prüfen, ob Solver verfügbar ist
        if not solver.available():
            raise RuntimeError("CPLEX ist nicht verfügbar.")

    except Exception:
        solver = pulp.PULP_CBC_CMD(timeLimit=solver_timeout, msg=debug, threads=thread_count)
    return solver
