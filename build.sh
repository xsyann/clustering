#!/usr/bin/env bash
##
## build.sh
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:30:49 2014 xs_yann
## Last update Thu Jun  5 14:44:39 2014 xs_yann
##

sudo rm -rf build dist && sudo python setup.py py2app -A

# Build documentation
epydoc --conf docs/epydoc.conf
