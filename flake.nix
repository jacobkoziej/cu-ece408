{
  description = "The Cooper Union - ECE 408: Wireless Communications";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

        inherit (pkgs) lib;
        inherit (pkgs) python3;

        python3-pkgs = python3.withPackages (
          ps: with ps; [
            galois
            miss-hit
            numpy
            pytest
            scipy
          ]
        );

      in
      {
        devShells.default = pkgs.mkShellNoCC (
          let
            pre-commit-bin = "${lib.getBin pkgs.pre-commit}/bin/pre-commit";

          in
          {
            packages = with pkgs; [
              black
              mdformat
              pre-commit
              python3-pkgs
              ruff
              shfmt
              toml-sort
              treefmt
              yamlfmt
              yamllint
            ];

            shellHook = ''
              ${pre-commit-bin} install --allow-missing-config > /dev/null
            '';
          }
        );

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
