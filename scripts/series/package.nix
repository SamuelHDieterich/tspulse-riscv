# The series selection command, built alone so it never shares an environment with the model
{
  lib,
  buildPythonApplication,
  hatchling,
  pandas,
  typer,
}:
buildPythonApplication {
  pname = "series";
  version = "0.1.0";
  pyproject = true;

  # Only what the build reads, so editing anything else doesn't rebuild it
  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./src
    ];
  };

  build-system = [ hatchling ];
  dependencies = [
    pandas # reads the CSVs
    typer # the command line
  ];
}
