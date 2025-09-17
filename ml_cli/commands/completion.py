import rich_click as click

@click.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh', 'fish']))
def completion(shell):
    """Show the shell completion setup instructions."""
    prog_name = 'ml-cli'
    if shell == 'bash':
        click.echo(f'# Add the following to your .bashrc or .bash_profile:')
        click.echo(f'eval "$({prog_name.upper().replace("-", "_")}_COMPLETE=bash_source {prog_name})"')
    elif shell == 'zsh':
        click.echo(f'# Add the following to your .zshrc:')
        click.echo(f'eval "$({prog_name.upper().replace("-", "_")}_COMPLETE=zsh_source {prog_name})"')
    elif shell == 'fish':
        click.echo(f'# Add the following to your config.fish:')
        click.echo(f'eval (env {prog_name.upper().replace("-", "_")}_COMPLETE=fish_source {prog_name})')
