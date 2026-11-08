# The export-weights command, built alone so it never shares an environment with the model
{
  lib,
  buildPythonApplication,
  hatchling,
  typer,
}:
buildPythonApplication {
  pname = "weights";
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
    typer # the command line
  ];
}
