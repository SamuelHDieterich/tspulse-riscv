{
  perSystem =
    { pkgs, ... }:
    let
      python = pkgs.python312;

      # uv builds these commands, so they live in a per-worktree venv. Adding that venv to PATH would expose its python too.
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

      score-baseline = pkgs.writeShellApplication {
        name = "score-baseline";
        runtimeInputs = with pkgs; [
          git # finds the worktree root, and records its commit with each result
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" score-baseline "$@"
        '';
      };

      capture-reference = pkgs.writeShellApplication {
        name = "capture-reference";
        runtimeInputs = with pkgs; [
          git # finds the worktree root
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" capture-reference "$@"
        '';
      };

      compare-reference = pkgs.writeShellApplication {
        name = "compare-reference";
        runtimeInputs = with pkgs; [
          git # finds the worktree root
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" compare-reference "$@"
        '';
      };

      prepare-rows = pkgs.writeShellApplication {
        name = "prepare-rows";
        runtimeInputs = with pkgs; [
          git # finds the worktree root
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" prepare-rows "$@"
        '';
      };

      check-model = pkgs.writeShellApplication {
        name = "check-model";
        runtimeInputs = with pkgs; [
          git # finds the worktree root
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" check-model "$@"
        '';
      };

      check-spike = pkgs.writeShellApplication {
        name = "check-spike";
        runtimeInputs = with pkgs; [
          git # finds the worktree root, and records its commit with each result
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" check-spike "$@"
        '';
      };

      benchmark = pkgs.writeShellApplication {
        name = "benchmark";
        runtimeInputs = with pkgs; [
          git # finds the worktree root, and records its commit with each result
          uv # runs the command from its venv
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" benchmark "$@"
        '';
      };

      time-host = pkgs.writeShellApplication {
        name = "time-host";
        runtimeInputs = with pkgs; [
          git # finds the worktree root, and records its commit with each result
          uv # runs the command from its venv
          time # GNU time, which measures each C process's peak memory
        ];
        text = /* bash */ ''
          exec uv run --locked --project "$(git rev-parse --show-toplevel)/scripts/score" time-host "$@"
        '';
      };
    in
    {
      devShells.python = pkgs.mkShell {
        packages = with pkgs; [
          python # runs the original model
          uv # manages the virtualenv and the lock file
          score-series # scores one series with the original model
          score-baseline # scores many series with IBM's metric, for the accuracy baseline
          capture-reference # saves every inner output on fixed windows, to check the C port against
          compare-reference # shows where a dump first differs from the reference
          prepare-rows # turns a series into the row file the C program reads
          check-model # runs the C program's dump build on the reference windows and compares it
          check-spike # runs windows on the host and under Spike and checks they write the same bytes
          benchmark # runs a labelled build's accuracy windows and counts its instructions under Spike
          time-host # times the C builds against the PyTorch pipeline on this computer
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
