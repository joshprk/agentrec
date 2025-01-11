{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    lib = nixpkgs.lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    getPkgs = system: import nixpkgs {inherit system;};
  in {
    devShells = forAllSystems (system: let
      pkgs = getPkgs system;
    in {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3
        ];

        shellHook = ''
          [ ! -d "./venv" ] && python -m venv venv
          source venv/bin/activate
        '';
      };
    });

    packages = forAllSystems (system: let
      pkgs = getPkgs system;
    in rec {
      default = agentrec;
      agentrec = pkgs.python3Packages.buildPythonPackage {
        pname = "agentrec";
        version = "0.1.0";
        src = ./agentrec;

        buildInputs = with pkgs; [
          python3
        ];

        shellHook = ''
          [ ! -d "./venv" ] && python -m venv venv
          source venv/bin/activate
        '';
      };
    });

    formatter = forAllSystems (system: let
      pkgs = getPkgs system;
    in
      pkgs.alejandra);
  };
}
