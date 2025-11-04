# CPE727-2025-03/Experimentos/DeepNN
Pasta a ser utilizada para o desenvolvimento dos experimentos desenvolvidos na matéria no tema DeepNN

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[contributors-url]: https://github.com/natmourajr/CPE727-2025-03/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[forks-url]: https://github.com/natmourajr/CPE727-2025-03/network/members
[stars-shield]: https://img.shields.io/github/stars/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[stars-url]: https://github.com/natmourajr/CPE727-2025-03/stargazers
[issues-shield]: https://img.shields.io/github/issues/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[issues-url]: https://github.com/natmourajr/CPE727-2025-03/issues
[license-shield]: https://img.shields.io/github/license/natmourajr/CPE727-2025-03.svg?style=for-the-badge
[license-url]: https://github.com/natmourajr/CPE727-2025-03/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/natanael-moura-junior-425a3294

## Para utilizar:

_Abaixo seguem as instruções de utilização do repositório_


1. Monte a imagem Docker na sua máquina
```bash
docker build . --tag=natmourajr/deepnn:lastest --no-cache
```
Obs: `--no-cache` é pra limpar o cache e reiniciar o build

2. Rode a imagem Docker na sua máquina
```bash
docker run --rm -it -v $(pwd):/workspace natmourajr/deepnn:lastest
```

Obs: Caso esteja no Windows, rode o código abaixo
```bash
docker run --rm -it -v ${pwd}:/tf/workspace -p 8880:8888 natmourajr/deepnn:lastest 
```

Obs: Caso a extração da máquina dê um erro do tipo "no space left on device"
```bash

```

