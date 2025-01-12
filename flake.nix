{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    mach-nix = {
      url = "github:davhau/mach-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    lib = nixpkgs.lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    getPkgs = system: import nixpkgs {inherit system;};
  in {
    packages = forAllSystems (system: {
      agentrec = mach-nix.buildPythonPackage {
        pname = "agentrec";
        src = ./agentrec;
        requirements = builtins.readFile ./requirements.txt;
      };

      default = self.packages.${system}.agentrec;
    });

    formatter = forAllSystems (system: let
      pkgs = getPkgs system;
    in
      pkgs.alejandra);
  };
}
