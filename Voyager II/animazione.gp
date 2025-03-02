# define fixed axis-ranges
set xrange [-6e12:6e12]
set yrange [-6e12:6e12]

# filename and n=number of lines of your data 
filedata = 'object1.dat'
n = system(sprintf('cat %s | wc -l', filedata))

do for [j=1:n] {
    set title 'giorno '.j
    plot 'object1.dat'  u 2:3 every ::1::j w l lw 2, \
          'object1.dat' u 2:3 every ::j::j w p pt 7 ps 2, \
           'object2.dat'  u 2:3 every ::1::j w l lw 2, \
          'object2.dat' u 2:3 every ::j::j w p pt 7 ps 2, \
'object3.dat'  u 2:3 every ::1::j w l lw 2, \
          'object3.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object5.dat'  u 2:3 every ::1::j w l lw 2, \
          'object5.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object6.dat'  u 2:3 every ::1::j w l lw 2, \
          'object6.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object11.dat'  u 2:3 every ::1::j w l lw 2, \
          'object11.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object17.dat'  u 2:3 every ::1::j w l lw 2, \
          'object17.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object22.dat'  u 2:3 every ::1::j w l lw 2, \
          'object22.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'object24.dat'  u 2:3 every ::1::j w l lw 2, \
          'object24.dat' u 2:3 every ::j::j w p pt 7 ps 2,\
'VoyagerIInasa.dat'  u 2:3 every ::1::j w l lw 2, \
          'VoyagerIInasa.dat' u 2:3 every ::j::j w p pt 7 ps 2



    pause 0.000007
}

