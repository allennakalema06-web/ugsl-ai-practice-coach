"""CI release gate: fail if configured PostgreSQL coverage silently skips."""

import argparse
import os
import xml.etree.ElementTree as ET


def verify(path):
    if not os.environ.get('UGSL_TEST_DATABASE_URL', '').strip():
        raise ValueError('CI requires disposable PostgreSQL configuration')
    root = ET.parse(path).getroot()
    cases = [c for c in root.iter('testcase') if '.postgres.' in c.get('classname', '') or
             c.get('classname', '').startswith('tests.postgres') or c.get('classname', '').startswith('postgres.')]
    if len(cases) < 121:
        raise ValueError('PostgreSQL suite was not fully collected')
    if any(c.find('skipped') is not None or c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('PostgreSQL release gate contains skipped or failed tests')
    print(f'PostgreSQL CI gate passed: {len(cases)} cases executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('junit_xml')
    args = parser.parse_args()
    verify(args.junit_xml)
