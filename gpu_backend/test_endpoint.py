#!/usr/bin/env python3
import os, time, requests
url=os.environ["GPU_API_URL"].rstrip("/")
headers={"Authorization": "Bearer " + os.environ["GPU_API_KEY"]}
r=requests.get(url+"/health", headers=headers, timeout=30)
r.raise_for_status();print("Health:",r.json())
r=requests.post(url+"/jobs",headers=headers,json={"note":"Pump PX-102 vibrated twice after startup. Pressure was 4.3 bar. No bearing temperature measurement was made. The operator stopped the pump for inspection.","stages":["SFT"],"max_new_tokens":128},timeout=30)
r.raise_for_status();job=r.json()["job_id"];print("Job:",job)
for _ in range(150):
  r=requests.get(url+"/jobs/"+job,headers=headers,timeout=30)
  r.raise_for_status();data=r.json()
  if data["status"]!="running":
    print("Status:",data["status"]);print(data.get("result",data.get("error")));break
  time.sleep(3)
else:print("Still processing; check RunPod console")
