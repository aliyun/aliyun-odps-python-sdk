#!/usr/bin/env python
import argparse
import os
import sys

from odps import ODPS
from odps.udf.tools import runners


def _chr_if_necessary(s):
    try:
        n = int(s)
        return chr(n)
    except ValueError:
        return s


def main():
    sys.path.insert(0, os.getcwd())

    parser = argparse.ArgumentParser(description="ODPS Python UDF tools")
    parser.add_argument(
        "-D",
        metavar="delim",
        type=str,
        default=",",
        help="Line delimiter that separates lines into columns, default is \",\"",
    )
    parser.add_argument(
        "-N", metavar="null", type=str, default="NULL", help="NULL indicator"
    )
    parser.add_argument(
        "-I",
        metavar="stdin",
        type=str,
        default="sys.stdin",
        help="standard input, sys.stdin as default",
    )
    parser.add_argument(
        "-t",
        "--table",
        metavar="table",
        type=str,
        help="table name, can also specify partitions and columns like table.p(p1=1,p2=2).c(c1,c2)",
    )
    parser.add_argument("--project", metavar="project", type=str, help="project name")
    parser.add_argument(
        "--access-id", metavar="access_id", type=str, help="access id of ODPS"
    )
    parser.add_argument(
        "--secret-access-key",
        metavar="secret_access_key",
        type=str,
        help="access key of ODPS",
    )
    parser.add_argument(
        "--endpoint", metavar="endpoint", type=str, help="endpoint of ODPS"
    )
    parser.add_argument(
        "--record-limit",
        metavar="record_limit",
        type=int,
        default=None,
        help="limitation of records",
    )
    parser.add_argument(
        "clz",
        metavar="your_script.class_name",
        type=str,
        help="The full import path of your UDF class",
    )
    args = parser.parse_args()

    delim = _chr_if_necessary(args.D)
    null_indicator = _chr_if_necessary(args.N)

    table_desc = args.table
    access_id = args.access_id
    secret_access_key = args.secret_access_key
    project = args.project
    endpoint = args.endpoint
    record_limit = args.record_limit

    # Import user class
    pkg, name = args.clz.rsplit(".", 1)
    usermod = __import__(pkg, globals(), locals(), [name])
    clz = getattr(usermod, name)

    # get stdin
    pkg, name = args.I.rsplit(".", 1)
    usermod = __import__(pkg, globals(), locals(), [name])
    stdin = getattr(usermod, name)

    if table_desc and access_id and secret_access_key:
        odps_entry = ODPS(access_id, secret_access_key, project, endpoint)
        udf_runner = runners.get_table_runner(clz, odps_entry, table_desc, record_limit)
    else:
        udf_runner = runners.get_csv_runner(clz, delim, null_indicator, stdin)
    udf_runner.run()


if __name__ == "__main__":
    main()
