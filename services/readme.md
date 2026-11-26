# Services — Box Archive

Folder ini berisi layanan-layanan pemrosesan inti (*microservices* / modul internal) untuk sistem **Box Archive**:

- [`ai_pipeline/`](./ai_pipeline/README.md): Modul **Speech-to-Text & Summarization (Audio & NLP Pipeline)** yang mengubah rekaman audio kelas berbahasa Indonesia menjadi transkrip utuh dan rangkuman terstruktur 3 segmen (*poin utama materi*, *sesi tanya jawab*, dan *tugas*), serta menghapus audio mentah secara otomatis setelah arsip berhasil disimpan.
