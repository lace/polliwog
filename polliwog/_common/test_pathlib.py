import simplejson as json

from .pathlib import SCHEMA_PATH, project_relative_path


def test_schema_matches_types_package():
    with open(SCHEMA_PATH, "r") as f:
        schema_from_project = json.load(f)

    with open(
        project_relative_path("types", "src", "generated", "schema.json"), "r"
    ) as f:
        schema_from_types_package = json.load(f)

    assert schema_from_project == schema_from_types_package
