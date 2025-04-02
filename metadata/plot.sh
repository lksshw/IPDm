#!/bin/bash
set -e

# Script to replicate figures from the main text.
#
#
# Fetches cached simulation files, stores them in the ./data/ folder
# runs plot scripts on these files and stores them in the ./figures folder

echo "----"
echo "Running [$(date)]"
echo "----"

echo "**** setting up storage directories [$(date)]"

currdir=$(pwd)'/'

echo "**** script running at $currdir [$(date)]"

# download files
if ! [ -f "${currdir}/data.zip" ]; then
	echo "save data not found";
	echo "**** downloading simulation files [$(date)] ...";
	wget "https://www.dropbox.com/scl/fi/zkwe96l96x09tyyvhgpel/data.zip?rlkey=7tepqg6ql752s7g3x2fsp69cc&st=0xsz3uz1&dl=0" -O data.zip;
fi

# unzip to ./data
if ! unzip -o data.zip; then
	echo "corrupted data";
	echo "**** downloading simulation files [$(date)] ...";
	wget "https://www.dropbox.com/scl/fi/zkwe96l96x09tyyvhgpel/data.zip?rlkey=7tepqg6ql752s7g3x2fsp69cc&st=0xsz3uz1&dl=0" -O data.zip;
fi

echo "**** plotting [$(date)] ..."

echo "* figure 2"
python3 plot_scripts/fig2.py

echo "* figure 3"
python3 plot_scripts/fig3.py

echo "* figure 4"
python3 plot_scripts/fig4.py

echo "* figure 5"
python3 plot_scripts/fig5.py

echo "* figure 6"
python3 plot_scripts/fig6.py

echo "* figure 8"
python3 plot_scripts/fig8.py

#remove data.zip
echo "**** deleting data.zip [$(date)]"
rm -f data.zip

echo "**** COMPLETE [$(date)]"








