#!/bin/bash
pkill -9 "iperf3"
pkill -9 "srsenb"
pkill -9 "srsepc"
pkill -9 "srsue"
pkill -9 $(ps aux | awk '/[p]ython3 \.\.\/multi_ue.py/{print $2}')
kill $(ps aux | awk '/port-forward /{print $2}')
pkill -9 "python3"
