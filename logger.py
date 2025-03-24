from datetime import datetime



class Logger:
    def __init__(self):
        self._process_start_time:dict[str,dict[str,datetime]] = {}
    
    def start_process(self,profile:str,process:str):
        self._process_start_time.setdefault(profile, {})

        if process in self._process_start_time[profile]:
            print(f"[{profile}] Warn: {process} is not ended.")

        now = datetime.now()
        self._process_start_time[profile][process] = now
        print(f"{now} [{profile}] {process} start")
    
    def end_process(self,profile:str,process:str):
        now = datetime.now()
        if not profile in self._process_start_time or not process in self._process_start_time[profile]:
            print(f"[{profile}] Warn: {process} is not started.")
            print(f"{now} [{profile}] {process} end.")
            return
        started = self._process_start_time[profile][process]
        del self._process_start_time[profile][process]
        print(f"{now} [{profile}] {process} end. It takes {now.timestamp()-started.timestamp()}s")

logger = Logger()