/**
 * Download the complete Quran with full tashkeel from alquran.cloud API.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const OUTPUT_DIR = __dirname;

function downloadSurah(surahNum) {
    return new Promise((resolve, reject) => {
        const url = `https://api.alquran.cloud/v1/surah/${surahNum}/quran-uthmani`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    console.log('='.repeat(60));
    console.log('Downloading Complete Quran with Tashkeel');
    console.log('='.repeat(60));

    const allSurahs = [];
    const allAyahs = [];

    for (let surahNum = 1; surahNum <= 114; surahNum++) {
        process.stdout.write(`\rDownloading Surah ${surahNum}/114...`);

        try {
            const data = await downloadSurah(surahNum);

            if (data && data.status === 'OK') {
                const surahData = data.data;
                allSurahs.push({
                    number: surahData.number,
                    name: surahData.name,
                    englishName: surahData.englishName,
                    englishNameTranslation: surahData.englishNameTranslation,
                    revelationType: surahData.revelationType,
                    numberOfAyahs: surahData.numberOfAyahs
                });

                for (const ayah of surahData.ayahs) {
                    allAyahs.push({
                        surah: surahNum,
                        ayah: ayah.numberInSurah,
                        text: ayah.text,
                        number: ayah.number,
                        juz: ayah.juz,
                        page: ayah.page
                    });
                }
            }
        } catch (e) {
            console.error(`\nError downloading surah ${surahNum}: ${e.message}`);
        }

        // Be nice to the API
        await sleep(100);
    }

    console.log('\n');

    // Save full JSON data
    const fullData = {
        surahs: allSurahs,
        ayahs: allAyahs,
        total_ayahs: allAyahs.length,
        total_surahs: allSurahs.length
    };

    const jsonPath = path.join(OUTPUT_DIR, 'quran_uthmani.json');
    fs.writeFileSync(jsonPath, JSON.stringify(fullData, null, 2), 'utf-8');
    console.log(`Saved JSON: ${jsonPath}`);

    // Save plain text (one ayah per line with reference)
    const txtPath = path.join(OUTPUT_DIR, 'quran_uthmani.txt');
    const txtContent = allAyahs.map(a => `${a.surah}:${a.ayah}\t${a.text}`).join('\n');
    fs.writeFileSync(txtPath, txtContent, 'utf-8');
    console.log(`Saved TXT: ${txtPath}`);

    // Save just the Arabic text (for easy processing)
    const arabicPath = path.join(OUTPUT_DIR, 'quran_arabic_only.txt');
    const arabicContent = allAyahs.map(a => a.text).join('\n');
    fs.writeFileSync(arabicPath, arabicContent, 'utf-8');
    console.log(`Saved Arabic-only: ${arabicPath}`);

    // Statistics
    console.log('\n' + '='.repeat(60));
    console.log('Download Complete!');
    console.log('='.repeat(60));
    console.log(`Total Surahs: ${allSurahs.length}`);
    console.log(`Total Ayahs: ${allAyahs.length}`);

    // Count words
    const totalWords = allAyahs.reduce((sum, a) => sum + a.text.split(/\s+/).length, 0);
    console.log(`Total Words: ${totalWords.toLocaleString()}`);

    // Count unique words
    const allWords = new Set();
    for (const ayah of allAyahs) {
        for (const word of ayah.text.split(/\s+/)) {
            allWords.add(word);
        }
    }
    console.log(`Unique Words: ${allWords.size.toLocaleString()}`);
}

main().catch(console.error);
