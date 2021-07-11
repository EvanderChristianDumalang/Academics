#!/usr/bin/env perl

use strict;
use warnings;

my $text = 'Menurut sebuah studi terbaru Kaspersky, anak-anak mengalami penurunan minat dalam game komputer pada beberapa bulan terakhir, terutama jika dibandingkan dengan periode sebelum terjadinya pandemi. Untuk Indonesia sendiri, minat online anak-anak terhadap game komputer mengalami perubahan signifikan dalam lima bulan pertama tahun ini. Berdasarkan statistik Kaspersky, minat tertinggi game komputer untuk anak-anak di negara ini berada di bulan April sebesar 16,32 persen dan menurun pada bulan Mei menjadi 13,23 persen. “Menurunnya minat pada game di komputer pribadi dapat dijelaskan oleh meningkatnya kebutuhan dalam penggunaannya untuk kegiatan lain. Misalnya, proses pendidikan lebih mudah di akses melalui komputer pribadi daripada di perangkat seluler," ujar Anna Larkina, pakar analisis konten web di Kaspersky, dalam keterangannya, Jumat, 5 Juni 2020. ';
my $max_words = <>;

my @words = split / /, $text;
my %counts;

for my $pos (0 .. $#words) {
  for my $phrase_len (0 .. ($pos >= $max_words ? $max_words - 1 : $pos)) {
    my $phrase = join ' ', @words[($pos - $phrase_len) .. $pos];
    $counts{$phrase}++;
  }
} 

use Data::Dumper;
print Dumper(\%counts);