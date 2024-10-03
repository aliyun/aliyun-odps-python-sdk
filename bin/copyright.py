# from https://github.com/sbrunner/hooks/blob/280356fe7906110b1a2275c553f2f40343a0e195/sbrunner_hooks/copyright.py
#  with minor modifications.
# licensed under BSD 2-Clause "Simplified" license
# Copyright (c) 2022-2024, Stéphane Brunner
"""Update the copyright header of the files."""

import argparse
import datetime
import os.path
import re
import subprocess  # nosec
import sys
from typing import TYPE_CHECKING, Tuple

import yaml

if TYPE_CHECKING:
    StrPattern = re.Pattern[str]
else:
    StrPattern = re.Pattern

CURRENT_YEAR = str(datetime.datetime.now().year)


def main() -> None:
    """Update the copyright header of the files."""
    args_parser = argparse.ArgumentParser("Update the copyright header of the files")
    args_parser.add_argument(
        "--config", help="The configuration file", default=".github/copyright.yaml"
    )
    args_parser.add_argument(
        "--required", action="store_true", help="The copyright is required"
    )
    args_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose mode"
    )
    args_parser.add_argument(
        "files", nargs=argparse.REMAINDER, help="The files to update"
    )
    args = args_parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)

    one_date_re = re.compile(
        config.get("one_date_re", r" Copyright (?P<year>[0-9]{4})")
    )
    two_date_re = re.compile(
        config.get("two_date_re", r" Copyright (?P<from>[0-9]{4})-(?P<to>[0-9]{4})")
    )
    one_date_format = config.get("one_date_format", " Copyright {year}")
    two_date_format = config.get("two_date_format", " Copyright {from}-{to}")
    year_re = re.compile(r"^(?P<year>[0-9]{4})-")
    license_file = config.get("license_file", "LICENSE")

    success = True
    no_git_log = False
    for file_name in args.files:
        try:
            status_str = subprocess.run(  # nosec
                ["git", "status", "--porcelain", "--", file_name],
                check=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
            ).stdout
            if status_str:
                used_year = CURRENT_YEAR
                if args.verbose:
                    print(f"File '{file_name}' is not committed.")
            else:
                if file_name == license_file:
                    date_str = subprocess.run(  # nosec
                        ["git", "log", "--pretty=format:%ci", "-1"],
                        check=True,
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                    ).stdout
                else:
                    date_str = subprocess.run(  # nosec
                        [
                            "git",
                            "log",
                            "--follow",
                            "--pretty=format:%ci",
                            "--",
                            file_name,
                        ],
                        check=True,
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                    ).stdout
                if not date_str:
                    if args.verbose:
                        print(f"No log found with git on '{file_name}'.")
                    else:
                        if not no_git_log:
                            print(
                                f"No log found with git on '{file_name}' (the next messages will be hidden)."
                            )
                            no_git_log = True
                    used_year = CURRENT_YEAR
                else:
                    if args.verbose:
                        print(f"File '{file_name}' was committed on '{date_str}'.")
                    used_year_match = year_re.search(date_str)
                    assert used_year_match is not None  # nosec
                    used_year = used_year_match.group("year")
        except FileNotFoundError:
            if not no_git_log:
                print("No Git found.")
                no_git_log = True
            used_year = CURRENT_YEAR
        except subprocess.CalledProcessError as error:
            print(f"Error with Git on '{file_name}' ({str(error)}).")
            used_year = CURRENT_YEAR

        with open(file_name, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
            file_success, content = update_file(
                content,
                used_year,
                one_date_re,
                two_date_re,
                one_date_format,
                two_date_format,
                file_name,
                args.required,
                args.verbose,
            )
        if not file_success:
            success = False
            with open(file_name, "w", encoding="utf-8") as file_obj:
                file_obj.write(content)
            print(f"Fixing copyright in '{file_name}'")

    if not success:
        sys.exit(1)


def update_file(
    content: str,
    last_year: str,
    one_date_re: StrPattern,
    two_date_re: StrPattern,
    one_date_format: str,
    two_date_format: str,
    filename: str = "<unknown>",
    required: bool = False,
    verbose: bool = False,
    current_year: str = CURRENT_YEAR,
) -> Tuple[bool, str]:
    """Update the copyright header of the file content."""
    two_date_match = two_date_re.search(content)
    if two_date_match:
        if two_date_match.group("from") == two_date_match.group("to"):
            if two_date_match.group("from") == current_year:
                return False, two_date_re.sub(
                    one_date_format.format(**{"year": current_year}), content
                )
            return (
                False,
                two_date_re.sub(
                    two_date_format.format(
                        **{"from": two_date_match.group("from"), "to": current_year}
                    ),
                    content,
                ),
            )

        if two_date_match.group("to") in (last_year, current_year):
            return True, content

        return False, two_date_re.sub(
            two_date_format.format(
                **{"from": two_date_match.group("from"), "to": current_year}
            ),
            content,
        )

    one_date_match = one_date_re.search(content)
    if one_date_match:
        copyright_year = one_date_match.group("year")

        if copyright_year == last_year:
            return True, content

        return False, one_date_re.sub(
            two_date_format.format(**{"from": copyright_year, "to": current_year}),
            content,
        )

    if required or verbose:
        print(f"No copyright found on '{filename}'.")
    return not required, content


if __name__ == "__main__":
    main()
