#!/usr/bin/env python3


import click
import boto3


@click.command()
@click.option('-r', '--role', type=str, required=True)
@click.option('-u', '--user', type=str, required=True)
@click.option('-t', '--token', type=str, required=True)
def main(role, user, token):
    client = boto3.client('sts')
    response = client.assume_role(
        RoleArn=f"arn:aws:iam::341687202595:role/{role}",
        RoleSessionName=f"{role}Session",
        DurationSeconds=43200,
        SerialNumber=f"arn:aws:iam::341687202595:mfa/{user}",
        TokenCode=token
    )

    access_key_id = response['Credentials']['AccessKeyId']
    secret_access_key = response['Credentials']['SecretAccessKey']
    session_token = response['Credentials']['SessionToken']

    access_key_id_cmd = f"AWS_ACCESS_KEY_ID={access_key_id}"
    secret_access_key_cmd = f"AWS_SECRET_ACCESS_KEY={secret_access_key}"
    session_token_cmd = f"AWS_SESSION_TOKEN={session_token}"

    env_file = open(".env", "w")
    env_file.write("AWS_SDK_LOAD_CONFIG=1\n")
    env_file.write(access_key_id_cmd + '\n')
    env_file.write(secret_access_key_cmd + '\n')
    env_file.write(session_token_cmd + '\n')
    env_file.close()


if __name__ == "__main__":
    main()
