def root_package_relative_path(*path_components):
    import os

    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", *path_components
    )


def project_relative_path(*path_components):
    return root_package_relative_path("..", *path_components)


SCHEMA_PATH = root_package_relative_path("schema.json")
