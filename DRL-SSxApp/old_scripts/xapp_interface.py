import socket
import json
import requests
import time
import subprocess
import re

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 3000

class KpmInterface:

    def __init__(self):
        self.current_state = list()
        self.current_ue = ""

    def get_kpms(self):
        print("Getting KPM: ", end="")
        recieved_data = list()
        start = time.time()
        recieved_data = list()
        while not recieved_data:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_HOST, SERVER_PORT))
                data = None
                while True:
                    time.sleep(0.1),
                    data = s.recv(1024)
                    if not data or data == b'':
                        break
                    else:
                        print(data)
                        recieved_data.append(json.loads(data))
                        response_message = "KPM recieved"
                        s.sendall(response_message.encode())
                s.close()
        
        print(recieved_data, time.time() - start)
        return recieved_data


class ConfInterface:

    def __init__(self):
        self.ues = []
        self.slices = []
        command = "sudo kubectl get svc -n ricxapp --field-selector metadata.name=service-ricxapp-drl-ss-rmr -o jsonpath='{.items[0].spec.clusterIP}'"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            raise ValueError(f"Error occured with kubectl: {result.stderr}")
        self.xapp = result.stdout.decode('utf-8')
        print("xapp: ", self.xapp)


    def get_slice(self, item):
        response = requests.get(f"http://{self.xapp}:8000/v1/slices/{item}")
        if response.status_code != 200:
            print(response.text)
            return None
        else:
            return response.json()

    def get_slices(self):
        response = requests.get(f"http://{self.xapp}:8000/v1/slices")
        if response.status_code != 200:
            return None
        else:
            return response.json()
       
    def get_ues(self):
        response = requests.get(f"http://{self.xapp}:8000/v1/ues")
        if response.status_code != 200:
            return None
        else:
            return response.json()
     
    def create_slice(prbs, slice_name):
        headers = {"Content-type": "application/json"}
        payload = {
            "name": slice_name,
            "allocation_policy": {
                "type": "proportional",
                "share": prbs
            }
        }
        response = requests.post(f"http://{self.xapp}:8000/v1/slices", headers=headers, json=payload)

        if response.status_code != 200:
            return False
        else:
            return True

    def bind_slice_to_eNB(self, nodeb, slice_name):
        response = requests.post(f"http://{self.xapp}:8000/v1/nodebs/{nodeb}/slices/{slice_name}")
        if response.status == 200:
            return True
        else:
            return False
    
    def create_ue(self, imsi):
        headers = {"Content-type": "application/json"}
        payload = { "imsi": imsi }
        response = requests.post(f"http://{self.xapp}:8000/v1/ues", headers=headers, json=payload)
        if response.status == 200:
            return True
        else:
            return False

    def bind_ue_to_slice(self, imsi, slice_name):
        response = requests.post(f"http://{self.xapp}:8000/v1/slices/{slice_name}/ues/{imsi}")
        if response.status_code == 200:
            return True
        else:
            return False

    def unbind_ue(self, imsi, slice_name):
        response = requests.delete(f"http://{self.xapp}:8000/v1/slices/{slice_name}/ues/{imsi}")

        if response.status_code == 200:
            return True
        print(response.text)
        return False


    def reallocate_prbs(self, prbs, slice_name):
       headers = {"Content-type": "application/json"}
       payload = { "name": slice_name, "allocation_policy": { "type": "proportional", "share": str(prbs) } }
       response = requests.put(f"http://{self.xapp}:8000/v1/slices", headers=headers, json=payload)

       if response.status_code != 200:
           return False
       else:
           return True

class IperfInterface:

    def __init__(self, namespaces):
        self.processes = dict()
        self.commands = dict()
        self.pattern = re.compile(r"[0-9]*\.[0-9]+ MB", re.IGNORECASE)
        self.namespcaces = namespaces
        for namespace in self.namespcaces:
            namespace_to_port = {
                "ue1":"5006",
                "ue2":"5020",
                "ue3":"5030"
            }

            namespace_to_bandwidth = {
                "ue1":"10M",
                "ue2":"10M",
                "ue3":"50M"
            }

            command = [
                "sudo",
                "ip",
                "netns", 
                "exec",
                namespace,
                "iperf3",
                "-c", 
                "172.16.0.1",
                "-p",
                namespace_to_port[namespace],
                "-t",
                "36000",
                "-i",
                "1",
                "-R",
                "-b",
                namespace_to_bandwidth[namespace]
            ]
            self.commands[namespace] = command

    def get_reading(self, namespace):
        print(f"Getting Iperf of {namespace}: ", end="")
        reading = None
        while reading is None:
            if self.processes[namespace].stderr:
                print(self.processes[namespace].stderr)
            for line in self.processes[namespace].stdout:
                reading = self.pattern.search(line.decode('utf-8').strip())
                if reading:
                    break
        return float(reading.group().split(" ")[0])

    def start(self):
        for namespace in self.namespcaces:
            if namespace in self.processes.keys():
                self.processes[namespace].terminate()
                self.processes[namespace].wait()
            new_process = subprocess.Popen(self.commands[namespace], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes[namespace] = new_process


    


if __name__ == "__main__":
    pass
