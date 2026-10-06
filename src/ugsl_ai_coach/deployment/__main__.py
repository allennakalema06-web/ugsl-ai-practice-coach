"""python -m ugsl_ai_coach.deployment: no infrastructure I/O for help/imports."""

import argparse

from ugsl_ai_coach.deployment.configuration import ROLES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Explicit UgSL production process composition')
    commands = parser.add_subparsers(dest='command', required=True)
    for role in ROLES:
        commands.add_parser(role, help=f'Run {role}; requires its runtime configuration')
    check = commands.add_parser('check-config', help='Validate configuration only; no infrastructure connection')
    check.add_argument('role', choices=ROLES)
    args = parser.parse_args(argv)
    try:
        from ugsl_ai_coach.core.config import Settings
        from ugsl_ai_coach.deployment.configuration import validate_config
        from ugsl_ai_coach.deployment.runtime import run_role
        settings = Settings()
        if args.command == 'check-config':
            validate_config(args.role, settings)
            print(f'Configuration valid: {args.role}')
        else:
            run_role(args.command, settings)
        return 0
    except Exception:
        # Includes validation/provider errors: no exception values or traceback.
        print('Startup failed: required configuration or runtime unavailable')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
