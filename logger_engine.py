import string
import random
import os
import glob
from datetime import datetime
from datetime import date
from textwrap import indent

class EmC_Logger:
    def __init__(self,folder_name:str,uid = None):

        if uid is None:
            letters = string.ascii_lowercase + string.digits
            self.uid = str(date.today())+"_["+"".join(random.choice(letters) for i in range(6))+"]"
            print(f"UUID: {self.uid}")
            
            self.dirname = f"{folder_name}/{self.uid}"
            self.logfilename = f"{self.dirname}/network_info.txt"
            self.netfoldername = f"{self.dirname}/network_para"
            os.makedirs(self.dirname,exist_ok=True)
            os.makedirs(self.netfoldername,exist_ok=True)
            print(f"logger created to {self.dirname}")
            self.write_msg(f"Initiated, file folder: {self.dirname}")
        else:
            self.dirname = glob.glob(f"{folder_name}/*{uid}*")[0]
            self.logfilename = f"{self.dirname}/network_info.txt"
            self.netfoldername = f"{self.dirname}/network_para"
            self.uid = self.dirname.split("/")[1]
    
    def write_msg(self,msg):
        with open(self.logfilename, "a") as logger:
            time_now = datetime.now().strftime("%m/%d/%y, %H:%M:%S")
            logger.write(f" @ {time_now}")
            logger.write("\n")
            
            msg = indent(msg, "    ")
            logger.write(msg)
            logger.write("\n\n")
            
    def print_msg(self,line = None):
        with open(self.logfilename,'r') as logger:
            logg = logger.read()
        if line is None:
            print(logg)
        else:
            logg = logg.split("@")
            if line>=len(logg): print("Line number out of range.")
            else: print("@" + logg[line])
    
    def get_ids(self):
        return (self.uid, self.dirname)