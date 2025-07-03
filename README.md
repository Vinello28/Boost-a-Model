# Boost-a-Model

[![License](https://img.shields.io/github/license/Vinello28/Boost-a-Model.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

## Overview

### Introduzione e Stato dell'Arte
Il Visual Servoing è una tecnica di controllo utilizzata nella robotica che guida i movimenti del robot in funzione delle informazioni visive. Gli approcci classici come Position-Based Visual Servoing (PBVS) e Image-Based Visual Servoing (IBVS) presentano limitazioni in termini di adattabilità e robustezza alle variazioni ambientali. Questo progetto confronta due approcci innovativi basati su deep learning.

### Approccio basato su Graph Neural Network (CNS)
**Concetto e Funzionamento del Modello:** il metodo CNS (Correspondence-encoded Neural Servoing) introduce una strategia di controllo basata sulla rappresentazione esplicita delle corrispondenze visive tra un'immagine corrente e una di riferimento, modellate come un grafo. In questo grafo, ciascun nodo rappresenta un keypoint rilevato nelle immagini, mentre gli archi codificano relazioni locali derivate dalla prossimità spaziale o dalla similarità tra descrittori.

**Architettura del Modello:** l'architettura di CNS è composta da quattro moduli principali: estrazione e matching dei keypoint, costruzione del grafo, encoder GNN e decoder. Il cuore dell'architettura è rappresentato dal Graph Convolutional Gated Recurrent Unit (GConvGRU), un'estensione della classica GRU, rete ricorrente progettata per gestire l'evoluzione temporale delle informazioni, adattata a operare su dati strutturati come i grafi.

**Fine-Tuning del Modello:** il fine-tuning si è svolto per un totale di 50 epoche, con un batch size pari a 16 e l'uso del teacher forcing per le prime epoche. L'ottimizzazione avviene tramite l'algoritmo AdamW, con un learning rate iniziale pari a 
5⋅e−4, ridotto a 1⋅e−4 per favorire un fine-tuning più stabile e un weight decay pari a 1⋅e−4.

### Approccio basato su Vision Transformer (ViT-VS)
**Concetto e Funzionamento del Modello:** il secondo approccio analizzato si basa sull'utilizzo di un Vision Transformer (ViT), come modulo per l'estrazione di feature semantiche da immagini. Per questo scopo è stata adottata l'architettura DINOv2 che è pre-addestrata su un vasto dataset di 142 milioni di immagini.

**Architettura del Modello:** l'architettura di ViT-VS è composta da una sequenza di moduli indipendenti che interagiscono in un ciclo di controllo visuale basato su IBVS. I moduli principali includono l'estrazione delle feature tramite ViT, il matching e la selezione dei punti guida, l'aggregazione contestuale, la compensazione rotazionale iniziale, il controllo IBVS classico e la stabilizzazione con EMA.

**Modifiche Appportate:** sono state effettuate alcune modifiche strutturali all'architettura ViT-VS per renderla indipendente da ROS e Gazebo, eliminando il modulo di rotazione e adattando il pre-processamento dell'input per utilizzare video registrati da videocamera ad alta definizione.

### Risultati Principali
I risultati sperimentali mostrano che entrambi gli approcci superano significativamente i metodi classici in termini di robustezza e generalizzazione. CNS eccelle nella precisione del controllo e gestione temporale, mentre ViT-VS si distingue per semplicità implementativa e adattamento immediato a diversi modelli.

### Conclusioni e Discussioni
L'analisi comparativa tra CNS e ViT-VS mette in luce un bilanciamento netto tra complessità computazionale e prestazioni operative. ViT-VS, in particolare il modello ViTs14, offre un compromesso ottimale per applicazioni embedded real-time, garantendo un'elevata accuratezza con tempi di inferenza estremamente ridotti. CNS, con l'architettura basata su GNN e detector come AKAZE, mostra un comportamento reattivo in presenza di rumore e perturbazioni, ma i costi computazionali rappresentano un significativo ostacolo per applicazioni real-time in contesti embedded.

Per migliorare ulteriormente le prestazioni complessive e ampliare l'applicabilità dei modelli, si suggeriscono alcune direzioni di ricerca e sviluppo, tra cui l'ottimizzazione e la quantizzazione dei modelli ViT, l'integrazione di tecniche di pruning e distillazione per CNS, e lo sviluppo di pipeline ibride.


## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Modular training and evaluation pipeline for deep learning models.
- Support for multiple computer vision architectures, including transformers.
- Tools for dataset preprocessing and augmentation.
- Experiment tracking and reproducibility.
- Visualizations for model predictions and metrics.

