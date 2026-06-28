# Hybrid Anomaly Framework


## System Architecture & Overview:
In this custom lab, as Snort and traditional intrusion detection systems rely on signature pre-defined rules and known malicious strings, a custom neural network is paired with traditional signature-based detection. Raw network packets obtained from Wireshark and parsed into behavioral features via Scapy are evaluated by the neural network to capture signature-less anomalies.

## Technical stack:
**Tools**: Python (NumPy, Pandas), Scapy, Snort, Wireshark, Nmap, Medusa, Netcat.

**Environment**: Metasploitable 2 (Target Host), Ubuntu (IDS Sensor deployment), Kali Linux (Attacker Machine).

**Framework**: Metasploit Framework.

## Reconnaissance:
A comprehensive nmap scan was executed from Kali Linux to identify vulnerable ports and attack surfaces
<img src="nmap-sV.png" width="660" alt="nmap -sV">
