{
  perSystem =
    { pkgs, ... }:
    let
      python = pkgs.python312;
    in
    {
      devShells.python = pkgs.mkShell {
        packages = with pkgs; [
          python # runs the original model
          uv # manages the virtualenv and the lock file
        ];

        # inputsFrom forwards shellHook but not env, so these are exported here.
        # uv would otherwise download its own interpreter, which does not run on NixOS.
        shellHook = /* bash */ ''
          export UV_PYTHON="${python}/bin/python"
          export UV_PYTHON_DOWNLOADS=never
        '';
      };
    };
}
