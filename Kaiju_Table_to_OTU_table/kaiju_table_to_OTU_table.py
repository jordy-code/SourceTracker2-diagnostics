#!/usr/bin/ python
# -*- coding: utf-8 -*-
#--------------------------------------------------------------
#Author: Laura Sisk-Hackworth
#Purpose: Extract taxa counts from kaiju output table into csv files for each domain
# -*- coding: utf-8 -*-

#import packages
import pandas as pd
import glob
#make dictionary between file name and tsv file
#all kaiju .tsv files need to be in the same directory and there needs to be no other .tsv files in that directory
files = glob.glob("*.tsv")
#add files to dictionary
ISS = {}
for file in files:
    df = pd.read_csv(file, sep='\t', index_col= 'taxon_name', usecols = [ 'reads', 'taxon_name'])
    #put dfs into dictionary. Key is the sample name
    file=file[:10]
    ISS[file] = df
for name, df in ISS.items():
        df.columns = df.columns + "_" + name
#make a database of all values 
big_df = pd.concat(ISS.values(), axis=1, sort=True)
big_df_zeroes = big_df.fillna(0)
#make new column to match index, called taxon_string
big_df_zeroes['taxon_string'] = big_df_zeroes.index
big_df_zeroes.head(10)
#split string column into new df
taxa_df = big_df_zeroes["taxon_string"].str.split(";", expand = True) 
#name taxa columns
taxa_df.columns=('rem', 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'unk')
#remove rem column and unk column
taxa_df=taxa_df.drop(['rem', 'unk', ], axis=1)
#append to big_df_zeroes on index
df_taxa_reads = pd.concat([taxa_df, big_df_zeroes], axis=1, sort=False)
#drop taxon string column
df_taxa_reads=df_taxa_reads.drop('taxon_string', axis=1)
#change column names to just the identifier find and replace reads with nothing?
df_taxa_reads.columns = df_taxa_reads.columns.str.replace('reads_', '')
df_taxa_reads.columns = df_taxa_reads.columns.str.replace('_', '')
#sort columns
df_taxa_reads.head(10)
#export metagenome table
df_taxa_reads.to_csv('OTU_metagenome.csv')
#split into domain specific OTU tables, then export
df_archaea=df_taxa_reads[df_taxa_reads['domain']=='Archaea']
df_archaea.to_csv('OTU_archaea.csv')
df_bacteria=df_taxa_reads[df_taxa_reads['domain']=='Bacteria']
df_bacteria.to_csv('OTU_bacteria.csv')  
df_eukaryota=df_taxa_reads[df_taxa_reads['domain']=='Eukaryota']
df_eukaryota.to_csv('OTU_eukaryota.csv')
