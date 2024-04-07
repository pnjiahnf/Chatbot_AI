import nltk
from nltk.chat.util import Chat, reflections

pola = [
    ['(hai|halo|hey)', ['Halo!', 'Hai, ada yang bisa saya bantu?']],
    ['(saya|aku) ingin (.*)', ['Kenapa Anda ingin %2?', 'Anda ingin %2 karena apa?']],
    ['(saya|aku) (.*)', ['Mengapa Anda %2?', 'Apakah ada alasan mengapa Anda %2?']],
    ['siapa kamu?', ['Saya adalah bot sederhana.', 'Anda bisa memanggil saya bot.', 'Ini Bapak Budi.', 'Ini Ibu Budi.']],
    ['apa kabar?', ['Saya hanyalah sebuah program, jadi saya tidak punya perasaan.']],
    ['(.*) berapa (.*)(harga|biaya)(.*)', ['Maaf, saya tidak bisa memberikan informasi tentang harga.']],
    ['buwung apa tu man', ['Buwung puyuh']],
    ['toko sedang tutup', ['Maaf, toko sedang tutup. Silakan kembali pada jam operasional.']],
    ['(berapa|jam berapa) (buka|tutup) toko?', ['Toko buka dari jam 09:00 hingga 21:00 setiap hari.']],
    ['apa saja yang dijual di toko ini?', ['Kami menjual berbagai macam produk, termasuk pakaian, makanan, dan aksesori.']],
    ['bagaimana cara memesan produk?', ['Anda dapat memesan produk melalui situs web kami atau mengunjungi toko fisik kami.']],
    ['berapa lama pengiriman produk?', ['Waktu pengiriman tergantung pada lokasi Anda dan metode pengiriman yang dipilih.']],
    ['apakah ada diskon atau promosi saat ini?', ['Anda bisa cek di situs web kami atau mengunjungi toko untuk informasi lebih lanjut mengenai diskon dan promosi.']],
    ['apakah ada program loyalitas pelanggan?', ['Ya, kami memiliki program loyalitas pelanggan yang memberikan berbagai manfaat kepada anggota.']],
    ['apakah produk dijamin kualitasnya?', ['Kami berkomitmen untuk menyediakan produk berkualitas kepada pelanggan kami.']],
    ['bagaimana cara mengembalikan produk?', ['Anda dapat mengembalikan produk yang tidak sesuai dengan kebijakan pengembalian kami.']],
    ['apakah toko ini menerima pembayaran dengan kartu kredit?', ['Ya, kami menerima pembayaran dengan kartu kredit, debit, dan uang tunai.']],
    ['apakah toko ini menyediakan layanan pengiriman?', ['Ya, kami menyediakan layanan pengiriman untuk pesanan Anda.']],
    ['bisakah Anda membantu saya memilih produk?', ['Tentu, saya akan berusaha membantu Anda dalam memilih produk yang sesuai.']],
    ['bagaimana cara menghubungi layanan pelanggan?', ['Anda dapat menghubungi layanan pelanggan kami di nomor yang tertera di situs web kami.']],
    ['apakah produk-produk di toko ini bermerek?', ['Ya, kami menjual produk bermerek dan juga produk lokal.']],
    ['bisakah saya mendapatkan nomor telepon toko ini?', ['Tentu, nomor telepon toko kami adalah XXX-XXXX-XXXX.']],
    ['apakah ada jaminan garansi untuk produk yang dibeli?', ['Ya, kami memberikan jaminan garansi untuk produk-produk tertentu.']],
    ['apakah toko ini ramah lingkungan?', ['Kami berusaha untuk menjadi ramah lingkungan dalam operasional kami.']],
    ['bagaimana cara melacak status pengiriman pesanan?', ['Anda dapat melacak status pengiriman pesanan melalui situs web kami.']],
    ['bagaimana cara mendaftar menjadi anggota program loyalitas?', ['Anda dapat mendaftar menjadi anggota program loyalitas kami di toko fisik atau melalui situs web kami.']],
    ['apakah ada batas waktu untuk pengembalian produk?', ['Ya, ada batas waktu tertentu untuk pengembalian produk.']],
    ['apakah ada biaya tambahan untuk pengiriman?', ['Biaya pengiriman tergantung pada lokasi pengiriman dan metode pengiriman yang dipilih.']],
    ['apakah toko ini menyediakan layanan bantuan instalasi produk?', ['Ya, kami menyediakan layanan bantuan instalasi untuk beberapa produk tertentu.']],
    ['bisakah saya memesan produk yang sedang kosong?', ['Maaf, produk yang sedang kosong tidak dapat dipesan.']],
    ['apakah toko ini menerima pembayaran dengan transfer bank?', ['Ya, kami menerima pembayaran dengan transfer bank.']],
    ['apakah ada kartu hadiah yang tersedia untuk pembelian?', ['Ya, kami memiliki kartu hadiah yang tersedia untuk pembelian.']],
    ['bisakah saya mengganti atau membatalkan pesanan?', ['Anda mungkin bisa mengganti atau membatalkan pesanan jika belum diproses.']],
    ['apakah produk-produk dijamin keamanannya?', ['Ya, kami memastikan produk-produk kami aman digunakan.']],
    ['apakah ada layanan pemesanan online?', ['Ya, Anda dapat melakukan pemesanan online melalui situs web kami.']],
    ['apakah toko ini menyediakan layanan bantuan teknis?', ['Ya, kami menyediakan layanan bantuan teknis untuk produk-produk tertentu.']],
    ['apakah ada potongan harga untuk pembelian dalam jumlah besar?', ['Ya, ada potongan harga untuk pembelian dalam jumlah besar.']],
    ['apakah ada persyaratan khusus untuk pengembalian produk?', ['Ya, ada persyaratan khusus yang perlu dipenuhi untuk pengembalian produk.']],
    ['apakah produk-produk dijamin keasliannya?', ['Ya, kami menjamin keaslian produk-produk kami.']],
    ['apakah produk-produk di toko ini tahan air?', ['Beberapa produk kami mungkin tahan air, namun tidak semuanya.']],
    ['bisakah saya mengajukan pertanyaan lebih lanjut tentang produk tertentu?', ['Tentu, saya akan berusaha membantu menjawab pertanyaan Anda tentang produk tertentu.']],
    ['apakah toko ini memiliki kebijakan privasi?', ['Ya, kami memiliki kebijakan privasi yang melindungi informasi pribadi pelanggan kami.']],
    ['apakah toko ini menerima pembayaran dengan PayPal?', ['Ya, kami menerima pembayaran dengan PayPal.']],
    ['apakah toko ini memiliki program poin reward?', ['Ya, kami memiliki program poin reward untuk pelanggan setia kami.']],
    ['apakah produk-produk di toko ini memiliki masa garansi yang sama?', ['Tidak, masa garansi mungkin berbeda-beda untuk setiap produk.']],
    ['apakah toko ini menerima pengembalian produk tanpa kemasan?', ['Kebijakan kami meminta produk dikembalikan dalam kondisi aslinya.']],
    ['bagaimana cara mendapatkan informasi tentang produk yang baru masuk?', ['Anda dapat mengunjungi situs web kami atau mengikuti akun media sosial kami untuk mendapatkan informasi tentang produk baru.']],
    ['apakah toko ini memiliki program referral?', ['Ya, kami memiliki program referral yang memberikan insentif kepada pelanggan yang merujuk orang lain ke toko kami.']],
    ['apakah toko ini memiliki layanan pelanggan 24/7?', ['Ya, kami memiliki layanan pelanggan 24/7 melalui telepon atau email.']],
    ['apakah toko ini menerima pembayaran dengan cryptocurrency?', ['Tidak, saat ini kami belum menerima pembayaran dengan cryptocurrency.']],
    ['bagaimana cara mendapatkan faktur atau kwitansi pembelian?', ['Anda akan mendapatkan faktur atau kwitansi pembelian bersama dengan produk yang Anda pesan.']],
    ['apakah toko ini menyediakan layanan pemasangan untuk produk-produknya?', ['Ya, kami menyediakan layanan pemasangan untuk beberapa produk tertentu.']]
]

chatbot = Chat(pola, reflections)

def mulai_chat():
    print("Selamat datang! Silakan mulai berbicara dengan bot. Ketik 'keluar' untuk mengakhiri percakapan.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == 'keluar':
            break
        else:
            response = chatbot.respond(user_input)
            print("Bot:", response)

mulai_chat()