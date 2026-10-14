"""Alembic environment: online migrations against NEPKEV_DATABASE_URL, or the URL a caller set on the config."""

from logging.config import fileConfig

from alembic import context

from nepkev.service.db import Base, make_engine
from nepkev.service.settings import Settings

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

url = config.get_main_option("sqlalchemy.url") or Settings().database_url
engine = make_engine(url)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
