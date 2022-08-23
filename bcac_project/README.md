# BCAC_project

#### Description
1、Usage Introduction of the BCAC project
   The BCAC project contains two submodules: BCAC_pcap and BCAC_capture.

(1)BCAC_pcap: Used to analyze pcap files, can be run on windows or linux systems  
	Windows Compilation Conditions: vscode + mingw 10.2.0 + cmake  
	ubuntu Compilation conditions: vscode + gcc 9.3.0 + cmake
	Modify the settings in data.cfg   

	``
        BCAC_in_pcap_file = "/home/cayman/data/pcap/202006031400.pcap.cut_type0_0-20_payload0.pcap"
        BCAC_random_seed = 2022
        ```

(2)BCAC_capture: Used to analyze the captured data, running on linux systems due to the use of libpcap underlay.  
	ubuntu Compilation conditions: vscode + gcc 9.3.0 + cmake
	Modify the settings in data.cfg  

	```
        BCAC_dev = "enp0s31f6"
        BCAC_out_pcap_file = "/home/cayman/data/pcap/20210501_3.pcap"
        BCAC_dump_type = 2; //0 -- not dump, 1 -- only sample, 2 -- al
        BCAC_max_packet = 5000000
        BCAC_capture_time = 900
        ```

2、The data playback method used in the paper
![Image text](/images/system.png)  
<center>System Topology</center>  
<br>

Device Name	| Device Configuration
:---: | :---:
Host1 for Replay Traffic<br>Host2 for Replay Traffic<br>Traffic Analysis System | CPU: Inter Core i7-12700K 3.6GHz<br>RAM: 128GB 3200MHz<br>ROM: 5TB<br>NIC1: 10000Mbps<br>NIC2: 1000Mbps
TAP Switch | Centec V580-48X6Q-TAP

Details of data playback method:

-The traffic replay module includes steps such as pcap parsing, packet slicing, IP modification, checksum modification, topology reconstruction and replay code. (1)The "pcap parsing" is responsible for the statistics of the collected normal traffic and attack traffic packet files respectively, removing the useless or unrepresentative link packets, leaving N links, and obtaining their source IP list and destination IP list. (2) The "packet splitting " refers to removing packets whose source or destination IPs are not in the IP list parsed in the first step. (3) Modify the source and destination IPs to Mininet intranet environment IPs respectively, while modifying the checksum to make the packets replayable. (4) Finally, according to the source and destination IP lists, generate the corresponding mininet topology and replay code, and run the traffic packet environment on the two high-performance hosts to simulate the real network respectively.

-The traffic collection module uses Centec's V580 TAP series 10GbE collection and shunting device, and after configuring the flow table, the real traffic packets simulated by the two high-performance hosts are copied to the traffic analysis server for subsequent feature extraction and situational awareness analysis.

-We set up the corresponding sampling rate and its feature extraction module. The feature extraction module samples the input traffic according to the algorithm requirements, and then calculates the features according to the feature extraction algorithm and stores them in the corresponding Redis database, which is a high-speed memory database that meets the high concurrency requirements in real time and is suitable for the high traffic feature access scenario of this system.


3、Introduction to the dataset
  The experiment dataset DataSet1 contains 453,043,378 packets collected by the MAWI working group on a 10Gbps link on June 3, 2020（http://mawi.wide.ad.jp/mawi/samplepoint-G/2020/202006031400.html）.  
  We divide all packets into the training set Dtrain and the test set Dtest in a 2:1 ratio in chronological order and then divide the training set Dtrain into a training subset and a validation subset according to the ratio of 7:3.  
  To validate our proposed concept drift detector and batch updater, we collect DDoS traffic containing 141,059,271 
packets for simulating drift samples in a real network environment. We mix DDoS traffic packets and Dtest to simulate the subsequent reached traffic containing drift samples, which we call DataSet2. 
