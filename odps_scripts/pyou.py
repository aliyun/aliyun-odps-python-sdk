#!/usr/bin/env python

import os
import sys

from odps.udf.tools import runners


def _chr_if_necessary(s):
    try:
        n = int(s)
        return chr(n)
    except ValueError:
        return s


def main():
    sys.path.insert(0, os.getcwd())

    from odps import udf
    # Arguments parsing
    import argparse
    parser = argparse.ArgumentParser(description='ODPS Python UDF tools')
    parser.add_argument('-D', metavar='delim', type=str, default=',',
                        help='Line delimiter that separates lines into columns, '
                        'default is ","')
    parser.add_argument('-N', metavar='null', type=str, default='NULL',
                        help='NULL indicator')
    parser.add_argument('-I', metavar='stdin', type=str, default='sys.stdin',
                        help='standard input, sys.stdin as default')
    parser.add_argument('clz', metavar='your_script.class_name', type=str, help='The full import path of your UDF class')
    args = parser.parse_args()
    delim = _chr_if_necessary(args.D)
    null_indicator = _chr_if_necessary(args.N)

    # Import user class
    pkg, name = args.clz.rsplit('.', 1)
    usermod = __import__(pkg, globals(), locals(), [name])
    clz = getattr(usermod, name)

    # get stdin
    pkg, name = args.I.rsplit('.', 1)
    usermod = __import__(pkg, globals(), locals(), [name])
    stdin = getattr(usermod, name)

    udf_runner = runners.get_default_runner(clz, delim, null_indicator, stdin)
    udf_runner.run()


if __name__ == '__main__':
    main()
