{ inputs, ... }:
{
  imports = [ inputs.git-hooks.flakeModule ];

  flake-file.inputs.git-hooks = {
    url = "github:cachix/git-hooks.nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  perSystem =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    {
      pre-commit.settings.hooks = {
        nixfmt.enable = true;
        ruff.enable = true;
        ruff-format.enable = true;
        clang-format = {
          enable = true;
          # The default also covers JSON and JavaScript.
          types_or = lib.mkForce [ "c" ];
        };
      };

      # Only the hook, because pre-commit is a Python application and as a package its dependencies would reach PYTHONPATH.
      devShells.hooks = pkgs.mkShell {
        inherit (config.pre-commit.devShell) shellHook;
      };
    };
}
