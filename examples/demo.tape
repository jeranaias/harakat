# Harakat Demo - VHS Tape File
# Install VHS: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape

# Configuration
Output demo.gif
Set FontSize 18
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

# Title
Type "# Harakat - Arabic Diacritization"
Enter
Sleep 1s

# Clear and run demo
Type "python harakat.py 'بسم الله الرحمن الرحيم'"
Enter
Sleep 2s

Type "# Quran mode: 99.997% accuracy"
Enter
Sleep 1s

Type "python harakat.py 'الحمد لله رب العالمين'"
Enter
Sleep 2s

Type "# Modern Standard Arabic: 2.29% DER"
Enter
Sleep 1s

Type "python harakat.py 'ذهب الطالب الى المدرسة'"
Enter
Sleep 2s

Type "# Arabic proverb"
Enter
Sleep 500ms

Type "python harakat.py 'العلم نور والجهل ظلام'"
Enter
Sleep 2s

Type "# 62x smaller than SOTA models (6.7 MB)"
Enter
Sleep 2s

Type "# github.com/jeranaias/harakat"
Enter
Sleep 3s
