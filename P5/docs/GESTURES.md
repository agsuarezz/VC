# Tiro y talco

MediaPipe sigue hasta dos manos. El filtro de LeBron utiliza el movimiento de las palmas y, cuando está disponible, la clasificación de palma abierta.

- **Tiro.** Sube una mano desde el pecho hasta cerca de la cara como si lanzaras. Aparece un balón y se reproduce el sonido. Deja la otra mano baja si también se ve.
- **Talco.** Empieza con ambas manos abajo. Súbelas juntas y abre las palmas al llegar a la altura de la cara. Baja ambas para preparar otra nube. El ascenso de ambas manos tiene prioridad y tolera pérdidas breves de detección.

La pausa entre gestos es de 1,15 s. El tiro requiere una subida de al menos 7,5 % de la altura del fotograma. El talco requiere 9 % en ambas manos. Los efectos son originales y se reproducen mediante winsound en Windows.
