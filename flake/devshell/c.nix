{
  perSystem =
    { pkgs, ... }:
    {
      devShells.c = pkgs.mkShell {
        packages = with pkgs; [
          gcc # host C compiler
          gnumake # build driver
        ];
      };
    };
}
