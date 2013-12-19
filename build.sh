#!/bin/sh

fileparser=~/projects/chemtools-webapp/chemtools/fileparser.py
data=/home/chris/research/ML/logs

echo "Parsing Generated"
python $fileparser -f $data/sets/ > data.csv
echo "Parsing Generated TD"
python $fileparser -f $data/setsTD/ >> data.csv
echo "Parsing Other"
python $fileparser -f $data/good/ >> data.csv
echo "Parsing Nonbenzo"
python $fileparser -f $data/nonbenzo/ >> data.csv
echo "Done"
