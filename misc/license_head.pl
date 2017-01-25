#!/usr/bin/env perl

# this script add license head to python source file

use strict;
use warnings;

my $license = <<"EOF";
# Copyright 1999-2017 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EOF

my $tmp = "tmp.$$";
for my $input ( @ARGV) {
    open IN, "<", $input or warn "failed to open $input\n";
    open OUT, ">", $tmp or die "failed to open $tmp\n";
    my $line = <IN>;
    if ($line =~ m/^# -\*- coding:/) {
        print OUT $line;
        print OUT $license;
    } elsif ($line =~ m/^#!\/usr/) {
        print OUT $line;
        $line = <IN>;
        if ($line =~ m/^# -\*- coding:/) {
            print OUT $line;
            print OUT $license;
        } else {
            print OUT $license;
            print OUT $line;
        }
    } else {
        print OUT $license;
        print OUT $line;
    }
    while (<IN>) {
        print OUT $_;
    }
    close OUT;
    close IN;
    0 == system("mv $tmp $input") or die "failed to overwrite $input\n";
}
