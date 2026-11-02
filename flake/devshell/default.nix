{
  perSystem =
    { config, pkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        # Each context is a shell of its own, because mkShell cannot merge a package list across files.
        inputsFrom = with config.devShells; [
          c
          python
          riscv
        ];

        packages = [ pkgs.git ];

        # nix develop adds an rpath under the checkout to NIX_LDFLAGS, which is split on spaces, so a path with spaces breaks every host link.
        env.NIX_NO_SELF_RPATH = "1";

        # Worktrees nest at any depth, so data/ is found from the shared git directory, not from the checkout.
        # It sits beside worktrees/ in the bare layout and at the repository root in a plain clone.
        shellHook = /* bash */ ''
          common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
          if [ "$(git --git-dir="$common_dir" rev-parse --is-bare-repository)" = true ]; then
            export TSPULSE_DATA="$(realpath "$common_dir/../../data")"
          else
            export TSPULSE_DATA="$(realpath "$common_dir/../data")"
          fi
          unset common_dir
        '';
      };
    };
}
