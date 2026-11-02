{
  perSystem =
    { pkgs, ... }:
    let
      python = pkgs.python312;

      # uv builds this command, so it lives in a per-worktree venv. Adding that venv to PATH would expose its python too.
      score-series = pkgs.writeShellApplication {
        name = "score-series";
        runtimeInputs = with pkgs; [
          git # finds the worktree root
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" score-series "$@"
        '';
      };
    in
    {
      devShells.python = pkgs.mkShell {
        packages = with pkgs; [
          python # runs the original model
          uv # manages the virtualenv and the lock file
          score-series # scores one series with the original model
        ];

        # inputsFrom forwards shellHook but not env, so these are exported here.
        # uv would otherwise download its own interpreter, which does not run on NixOS.
        shellHook = /* bash */ ''
          export UV_PYTHON="${python}/bin/python"
          export UV_PYTHON_DOWNLOADS=never

          # Prebuilt wheels expect these system libraries, which NixOS has no default path for.
          # Without them, numpy and torch fail on libstdc++.so.6, and numpy's bundled libgfortran on libz.so.1.
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
