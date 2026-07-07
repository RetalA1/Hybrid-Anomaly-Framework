# Hybrid Anomaly Framework


## System Architecture & Overview:
In this custom lab, as Snort and traditional intrusion detection systems rely on signature pre-defined rules and known malicious strings, a custom neural network is paired with traditional signature-based detection. Raw network packets obtained from Wireshark and parsed into behavioral features via Scapy are evaluated by the neural network to capture signature-less anomalies.

## Technical stack:
**Tools**: Python (NumPy, Pandas), Scapy, Snort, Wireshark, Nmap, Medusa, Netcat.

**Environment**: Metasploitable 2 (Target Host), Ubuntu (IDS Sensor deployment), Kali Linux (Attacker Machine).

**Framework**: Metasploit Framework.

## Reconnaissance & Baseline feature intake:
A comprehensive nmap scan was executed from Kali Linux to identify vulnerable ports and attack surfaces, establishing network properties needed to train the anomaly detection model.

<img src="nmap -sV.png" width="450" alt="nmap -sV">

Figure 1: Comprehensive Nmap scan & results


## Exploitation & Dataset Generation:
Multiple penetration testing measures were executed against the target host, providing behavioral varation to test model boundaries.

<img src="ssh privilege escalation.png" width="550" alt="privilege escalation"> 

Figure 2: Successful local privilege escalation from standard user to root.

<img src="medusa exploit.png" width="350" alt="medusa exploit"> 

Figure 3: SSH credential brute-forcing simulating high frequency noise.

<img src="RCE exploit.png" width="350" alt="RCE exploit"> 

Figure 4: RCE via Apache Tomcat manager generating web payload traffic.

<img src="VSFTPD exploit.png" width="350" alt="VSFTPD exploit"> 

Figure 5: vsftpd 2.3.4 backdoor exploitation on port 21.

<img src="telnet port 1524.png" width="350" alt="telnet port 1524">  

Figure 6: Remote command shell exploitation via telnet bind shell.


## Defensive & Traditional alerts:
Raw network activity was captured to document and analyse how traditional signature rules register before undergoing feature extraction.

<img src="medusa exploit logs.png" width="350" alt="medusa logs">  

Figure 7: Signature alerts captured by running Medusa exploit for initial threat verification (High frequency noise).

<img src="nc port 6200.png" width="350" alt="nc port 6200"> <img src="nc port 6200 logs.png" width="350" alt="nc 6200 logs">  

Figure 8: Netcat command execution and traditional signature alert mapping.

<img src="wireshark logs.png" width="350" alt="wireshark logs">  

Figure 9: Packet analysis tracking raw data on the physical interface veifying feature parsing.


## AI Anomaly detection & Model validation:
Structured behavioral features were fed into the custom network layers to outline hidden anomalies that bypassed signature detections

<img src="model accuracy.png" width="550" alt="model accuracy">

Figure 10: Custom neural network optimization showcasing training metrics and loss minimization. The validation accuracy reached 92.47%, with a decision rate of 100% and a result of 10 True positives & 0 False negatives, this proves the mathematical stability of the forward and backward propagations built on NumPy matrix operations is able to detect all attacks missed by traditional signature-based detection by adding a stratified split to ensure a proper training to test set ratio.

<img src="stealth packets.png" width="550" alt="stealth packets">

Figure 11: Evaluation of the Scapy parsed features dataset by a custom Python detection using the network matrix through activation layers and trained weights, flagging behavioral anomalies that left zero traces in traditional logs.


## Incident & Detection Analysis:
By utilizing a hybrid approach to traditional detection by combining signature-based & anomaly behavioural detection, this framework reduces the restrictions of a traditonal detection system by using a NumPy matrix based neural network. 
