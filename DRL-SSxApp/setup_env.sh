#!/bin/bash

# Run this file to setup the zmq with one UE
SLEEP_TIME=2
SLEEP_TIME_MID=5
SLEEP_TIME_LONG=10
SLEEP_TIME_LONGER=30

run_bg() {
	echo $1
	nohup $1 >$2 &
}

run_bg_priority() {
	echo $1
	chrt -r 99 nohup $1 >$2 &
}

run_screen() {
	echo $1
	screen -L -Logfile $2 -S $3 -d -m $1 2>&1
}

exit_handler() {
	./kill_stuff.sh
}

mkdir -p logs

trap exit_handler SIGINT

kubectl -n ricxapp rollout restart deployment ricxapp-drl-ss
sleep ${SLEEP_TIME}

# Run the EPC
ip netns del ue1 || true
ip netns add ue1
ip netns del ue2 || true
ip netns add ue2
ip netns del ue3 || true
ip netns add ue3
ip netns list
run_screen "srsepc" "./logs/epc.log" "srsepc"

export E2NODE_IP=$(hostname -I | cut -f1 -d' ')
export E2NODE_PORT=5006
export E2TERM_IP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath='{.items[0].spec.clusterIP}')

sleep ${SLEEP_TIME}

# Start the eNodeB
#sleep ${SLEEP_TIME_LONG}
run_screen "srsenb --enb.n_prb=50 --enb.name=enb1 --enb.enb_id=0x19B --rf.device_name=zmq --rf.device_args='fail_on_disconnect=true,tx_port=tcp://*:2000,rx_port=tcp://localhost:2009,id=enb,base_srate=23.04e6' --ric.agent.remote_ipv4_addr=$E2TERM_IP --log.all_level=warn --ric.agent.log_level=debug --log.filename=stdout --ric.agent.local_ipv4_addr=$E2NODE_IP --ric.agent.local_port=$E2NODE_PORT --slicer.enable=1 --slicer.workshare=0" "./logs/enb.log" "srsenb"

sleep ${SLEEP_TIME_LONG}
run_screen "srsue --rf.device_name=zmq --rf.device_args='tx_port=tcp://*:2010,rx_port=tcp://localhost:2300,id=ue,base_srate=23.04e6' --usim.algo=xor --usim.imsi=001010123456789 --usim.k=00112233445566778899aabbccddeeff --usim.imei=353490069873310 --log.all_level=warn --log.filename=stdout --gw.netns=ue1" "./logs/ue1.log" "ue1"

sleep ${SLEEP_TIME}
run_screen "srsue --rf.device_name=zmq --rf.device_args='tx_port=tcp://*:2007,rx_port=tcp://localhost:2400,id=ue,base_srate=23.04e6' --usim.algo=xor --usim.imsi=001010123456780 --usim.k=00112233445566778899aabbccddeeff --usim.imei=353490069873310 --log.all_level=warn --log.filename=stdout --gw.netns=ue2" "./logs/ue2.log" "ue2"

sleep ${SLEEP_TIME}
run_screen "srsue --rf.device_name=zmq --rf.device_args='tx_port=tcp://*:2008,rx_port=tcp://localhost:2500,id=ue,base_srate=23.04e6' --usim.algo=xor --usim.imsi=001010123456781 --usim.k=00112233445566778899aabbccddeeff --usim.imei=353490069873310 --log.all_level=warn --log.filename=stdout --gw.netns=ue3" "./logs/ue3.log" "ue3"

sleep ${SLEEP_TIME_LONG}
chmod 755 ../multi_ue.py

run_screen "../multi_ue.py" "./logs/multi_ue.log" "multi_ue"

sleep ${SLEEP_TIME_LONG}
sleep ${SLEEP_TIME_LONG}
sleep ${SLEEP_TIME_LONG}

# Run the iperf3 server
run_screen "iperf3 -s -B 172.16.0.1 -p 5006 -i 1" "./logs/iperf_s1.log" "iperf_s1"
run_screen "iperf3 -s -B 172.16.0.1 -p 5020 -i 1" "./logs/iperf_s2.log" "iperf_s2"
run_screen "iperf3 -s -B 172.16.0.1 -p 5030 -i 1" "./logs/iperf_s3.log" "iperf_s3"

sleep ${SLEEP_TIME}

#only needed for one time deployment
export KONG_PROXY=$(kubectl get svc -n ricplt -l app.kubernetes.io/name=kong -o jsonpath='{.items[0].spec.clusterIP}')
export E2MGR_HTTP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2mgr-http -o jsonpath='{.items[0].spec.clusterIP}')
export APPMGR_HTTP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-appmgr-http -o jsonpath='{.items[0].spec.clusterIP}')
export E2TERM_SCTP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath='{.items[0].spec.clusterIP}')
export ONBOARDER_HTTP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-xapp-onboarder-http -o jsonpath='{.items[0].spec.clusterIP}')
export RTMGR_HTTP=$(kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-rtmgr-http -o jsonpath='{.items[0].spec.clusterIP}')
export SRS=$(realpath .)

#don't need to do this everytime, only when deploying
#curl -L -X POST "http://$KONG_PROXY:32080/onboard/api/v1/onboard/download" --header 'Content-Type: application/json' --data-binary "@drl-ss-onboard.url"
#curl -L -X GET "http://$KONG_PROXY:32080/onboard/api/v1/charts"
#curl -L -X POST "http://$KONG_PROXY:32080/appmgr/ric/v1/xapps" --header 'Content-Type: application/json' --data-raw '{"xappName": "drl-ss"}'

# View the Kebernetes logs
run_bg "kubectl logs -f -n ricxapp -l app=ricxapp-drl-ss" "./logs/kub.log"

# Enable port forwarding for KPM collection
run_bg "kubectl port-forward -n ricxapp $(kubectl get pods -n ricxapp | grep -o 'ricxapp[^ ]*') 3000:3000" "./logs/port_forward.log"

# Add slices and Ues
../zmqthreeue.sh
kill -SIGUSR1 $(ps aux | awk '/python3 \.\/agent.py/{print $2}')
echo "Script Finished"
wait
