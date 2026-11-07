"""Official DeepSeek profile used by the shared V5 extractor and V2 judge."""
import os


def configuration(profile, model):
    if (profile, model) != ('deepseek_official', 'dso_DSv41Flash'):
        raise ValueError('Unsupported public evaluation configuration.')
    return (
        {'base_url': 'https://api.deepseek.com', 'api_key': os.getenv('DEEPSEEK_API_KEY', '').strip()},
        'deepseek-flash',
        {'thinking': {'type': 'disabled'}},
    )
