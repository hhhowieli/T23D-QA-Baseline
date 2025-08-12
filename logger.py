import os
import time

class Logger():
    def __init__(self, fn_prefix):
        self.fn_prefix = fn_prefix
        self.log_file = f"{fn_prefix}_{self.logger_time_name()}.txt"
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            
        print(self)
    
    def logger_time_name(self):
        
        curr_t = time.localtime()
        
        y = "%d" % curr_t.tm_year
        m = "%02d" %  curr_t.tm_mon
        d =  "%02d" % curr_t.tm_mday
        h =  "%02d" % curr_t.tm_hour
        min = "%02d" %  curr_t.tm_min
        s = "%02d" %  curr_t.tm_sec

        return f"{y}{m}{d}{h}{min}{s}"

    def log(self, msg, verbos=True, end="\n"):
        if verbos:
            print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg+end)

    def __str__(self):
        return self.log_file

