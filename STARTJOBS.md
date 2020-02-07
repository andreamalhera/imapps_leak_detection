Start Jobs CIP
	 	 	 	
**Bash-Skript**
* Schreibe ein bash_skript.sh mit dem Pfad zu deinem Python Skript, das ausgeführt werden soll.
Lege Datei in dein home directory
Beispielinhalt:

"#!/bin/sh
cd /home/k/kunzes/IMapps/imapps-master
python3 simplest_autoencoder.py "

**Start job**
* sbatch –partition=All bash_skript.sh

**See all jobs**
* squeue

**See your jobs**
* squeue –user=<your_name>

**Cancel a Job**
* scancel <jobID>


Man bekommt von jedem Job im directory home eine slurm-<jobID>.out