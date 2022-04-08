def project_relative_path(*path_components):
    import os

    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", *path_components
    )


SCHEMA_PATH = project_relative_path("types", "src", "generated", "schema.json")
