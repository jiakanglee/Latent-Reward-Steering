import os
import numpy as np
import torch
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import time
from utils.utils import print_and_flush
from utils.utils import load_activation_mean
from utils.sae import SAE

def clustering_agglomerative(activations, n_clusters, args):
    """
    Perform Agglomerative Hierarchical clustering on normalized activations.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    # Initialize Agglomerative Clustering with Ward linkage
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    
    # Fit model
    cluster_labels = model.fit_predict(activations)
    
    # Compute cluster centers (not provided by the model)
    cluster_centers = np.zeros((n_clusters, activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(activations[mask], axis=0)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/agglomerative', exist_ok=True)
    model_save_path = f'results/vars/agglomerative/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    return cluster_labels, cluster_centers

def clustering_spherical_kmeans(activations, n_clusters, args):
    """
    Perform Spherical KMeans clustering using cosine similarity.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    # Initialize KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        random_state=42,
        verbose=1
    )

    activations_norm = activations / np.linalg.norm(activations, axis=1, keepdims=True)
    
    # Fit model on normalized data (for cosine similarity)
    cluster_labels = kmeans.fit_predict(activations_norm)
    
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Normalize cluster centers for cosine similarity
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / norms
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/spherical_kmeans', exist_ok=True)
    model_save_path = f'results/vars/spherical_kmeans/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': kmeans,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    print_and_flush(f"    Spherical KMeans clustering completed in {time.time() - start_time:.2f} seconds total")

    return cluster_labels, cluster_centers

def clustering_gmm(activations, n_clusters, args):
    """
    Perform Gaussian Mixture Model clustering with pilot-fit then fine-tune approach.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments containing clustering parameters
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    n_samples = activations.shape[0]
    
    # Use clustering parameters from args
    pilot_size = min(args.clustering_pilot_size, n_samples)
    pilot_n_init = args.clustering_pilot_n_init
    pilot_max_iter = args.clustering_pilot_max_iter
    full_n_init = args.clustering_full_n_init
    full_max_iter = args.clustering_full_max_iter

    print_and_flush(f"    GMM clustering with {n_clusters} clusters on {n_samples} samples...")
    
    if n_samples > pilot_size:
        # Step 1: Pilot fit on sub-sample
        print_and_flush(f"    Step 1: Pilot fit on {pilot_size} samples with n_init={pilot_n_init}, max_iter={pilot_max_iter}...")
        pilot_start = time.time()
        
        # Sub-sample
        pilot_indices = np.random.choice(n_samples, pilot_size, replace=False)
        pilot_data = activations[pilot_indices]
        
        # Pilot GMM with multiple initializations
        pilot_gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=pilot_n_init,
            max_iter=pilot_max_iter,
            verbose=1
        )
        pilot_gmm.fit(pilot_data)
        pilot_time = time.time() - pilot_start
        print_and_flush(f"    Pilot fit completed in {pilot_time:.2f} seconds")
        
        # Step 2: Fine-tune on full dataset with good initialization
        print_and_flush(f"    Step 2: Fine-tune on full {n_samples} samples with n_init={full_n_init}, max_iter={full_max_iter}...")
        finetune_start = time.time()
        
        # Create final GMM with initialization from pilot
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=full_n_init,
            max_iter=full_max_iter,
            weights_init=pilot_gmm.weights_,
            means_init=pilot_gmm.means_,
            precisions_init=pilot_gmm.precisions_,
            verbose=1
        )
        gmm.fit(activations)
        finetune_time = time.time() - finetune_start
        print_and_flush(f"    Fine-tune completed in {finetune_time:.2f} seconds")
        
    else:
        # For small datasets, use standard approach
        print_and_flush(f"    Small dataset ({n_samples} samples), using standard approach with n_init=10...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=10,
            verbose=1
        )
        gmm.fit(activations)
    
    # Get cluster assignments
    cluster_labels = gmm.predict(activations)
    
    # Use means as cluster centers
    cluster_centers = gmm.means_
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/gmm', exist_ok=True)
    model_save_path = f'results/vars/gmm/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': gmm,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    total_time = time.time() - start_time
    print_and_flush(f"    GMM clustering completed in {total_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers

def clustering_pca_kmeans(activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by KMeans clustering.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(activations.shape[0], activations.shape[1], 100)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(activations)
    
    # Apply KMeans to reduced data
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        random_state=42,
        verbose=1
    )
    cluster_labels = kmeans.fit_predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(activations[mask], axis=0)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/pca_kmeans', exist_ok=True)
    model_save_path = f'results/vars/pca_kmeans/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'kmeans': kmeans,
            'pca': pca,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    print_and_flush(f"    PCA+KMeans clustering completed in {time.time() - start_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers

def clustering_pca_gmm(activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by GMM clustering with pilot-fit then fine-tune approach.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments containing clustering parameters
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    n_samples = activations.shape[0]
    
    # Use clustering parameters from args
    pilot_size = min(args.clustering_pilot_size, n_samples)
    pilot_n_init = args.clustering_pilot_n_init
    pilot_max_iter = args.clustering_pilot_max_iter
    full_n_init = args.clustering_full_n_init
    full_max_iter = args.clustering_full_max_iter
    
    print_and_flush(f"    PCA+GMM clustering with {n_clusters} clusters on {n_samples} samples...")
    
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(activations.shape[0], activations.shape[1], 100)
    
    # Apply PCA
    print_and_flush(f"    Applying PCA to {n_components} components...")
    pca_start = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(activations)
    pca_time = time.time() - pca_start
    print_and_flush(f"    PCA completed in {pca_time:.2f} seconds")
    
    if n_samples > pilot_size:
        # Step 1: Pilot fit on sub-sample
        print_and_flush(f"    Step 1: Pilot fit on {pilot_size} samples with n_init={pilot_n_init}, max_iter={pilot_max_iter}...")
        pilot_start = time.time()
        
        # Sub-sample with fixed random state for reproducibility
        np.random.seed(42)
        pilot_indices = np.random.choice(n_samples, pilot_size, replace=False)
        pilot_reduced_data = reduced_data[pilot_indices]
        
        # Pilot GMM with multiple initializations
        pilot_gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=pilot_n_init,
            max_iter=pilot_max_iter,
            verbose=1
        )
        pilot_gmm.fit(pilot_reduced_data)
        pilot_time = time.time() - pilot_start
        print_and_flush(f"    Pilot fit completed in {pilot_time:.2f} seconds")
        
        # Step 2: Fine-tune on full dataset with good initialization
        print_and_flush(f"    Step 2: Fine-tune on full {n_samples} samples with n_init={full_n_init}, max_iter={full_max_iter}...")
        finetune_start = time.time()
        
        # Create final GMM with initialization from pilot
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=full_n_init,
            max_iter=full_max_iter,
            weights_init=pilot_gmm.weights_,
            means_init=pilot_gmm.means_,
            precisions_init=pilot_gmm.precisions_,
            verbose=1
        )
        gmm.fit(reduced_data)
        finetune_time = time.time() - finetune_start
        print_and_flush(f"    Fine-tune completed in {finetune_time:.2f} seconds")
        
    else:
        # For small datasets, use standard approach
        print_and_flush(f"    Small dataset ({n_samples} samples), using standard approach with n_init=10...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(reduced_data)
    
    # Get cluster assignments
    cluster_labels = gmm.predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(activations[mask], axis=0)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/pca_gmm', exist_ok=True)
    model_save_path = f'results/vars/pca_gmm/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'gmm': gmm,
            'pca': pca,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    total_time = time.time() - start_time
    print_and_flush(f"    PCA+GMM clustering completed in {total_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers

def clustering_pca_agglomerative(activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by Agglomerative clustering.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(activations.shape[0], activations.shape[1], 100)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(activations)
    
    # Apply Agglomerative clustering to reduced data
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = agg.fit_predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(activations[mask], axis=0)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/pca_agglomerative', exist_ok=True)
    model_save_path = f'results/vars/pca_agglomerative/{model_id}_layer{args.layer}_clusters{n_clusters}.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'agglomerative': agg,
            'pca': pca,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers
        }, f)
    
    print_and_flush(f"    PCA+Agglomerative clustering completed in {time.time() - start_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers

 
def clustering_sae_topk(activations, n_clusters, args, topk=3):
    """
    Perform clustering using a top-k sparse autoencoder.
    Follows the TinySAE implementation from https://github.com/JoshEngels/TinySAE/blob/main/tiny_sae.py
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters (also number of features in the SAE)
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
    topk : int
        Number of top activations to keep during training (default: 3)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers)
    """
    start_time = time.time()
    # Ensure we're working with torch tensors on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy array to torch tensor
    X = torch.from_numpy(activations).float().to(device)
    input_dim = activations.shape[1]
    
    # Initialize model, loss, and optimizer
    sae = SAE(input_dim, n_clusters, k=topk).to(device)
    # Auto-select LR using 1 / sqrt(d) scaling law from TinySAE
    lr = 2e-4 / (n_clusters / (2**14)) ** 0.5
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Train the autoencoder
    max_epochs = 300
    batch_size = min(512, activations.shape[0])  # Adjust batch size based on data size
    n_samples = X.shape[0]
    
    # Early stopping parameters
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print_and_flush(f"Training sparse autoencoder with {n_clusters} clusters, topk={topk}...")
    for epoch in range(max_epochs):
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        # Set model to training mode
        sae.train()
        total_loss = 0
        
        # Create mini-batches and train
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, n_samples)]
            batch_X = X[batch_indices]
            
            # Forward pass
            predicted = sae(batch_X)
            
            # Compute loss - using the Loss function from TinySAE
            error = predicted - batch_X
            loss = (error**2).sum()
            loss /= ((batch_X - batch_X.mean(dim=0, keepdim=True)) ** 2).sum()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply decoder normalization after each step (as in TinySAE)
            sae.set_decoder_norm_to_unit_norm()
            
            total_loss += loss.item() * len(batch_indices)

            # Assert total_loss is not nan
            if np.isnan(total_loss):
                raise ValueError(f"Total loss is nan at epoch {epoch}")
            
            # Free up memory
            del batch_X, predicted, error, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average loss
        avg_loss = total_loss / n_samples
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print_and_flush(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model state
            best_model_state = {
                'encoder_weight': sae.encoder.weight.data.clone(),
                'encoder_bias': sae.encoder.bias.data.clone(),
                'decoder_weight': sae.W_dec.data.clone(),
                'b_dec': sae.b_dec.data.clone()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_and_flush(f"Early stopping at epoch {epoch+1}")
                break
    
    # After training, restore the best model state found
    if best_model_state:
        print_and_flush(f"Restoring best model with loss {best_loss:.6f}")
        sae.encoder.weight.data = best_model_state['encoder_weight']
        sae.encoder.bias.data = best_model_state['encoder_bias']
        sae.W_dec.data = best_model_state['decoder_weight']
        sae.b_dec.data = best_model_state['b_dec']
    
    # Use the encoder to determine cluster assignments - get the highest activating feature for each example
    sae.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues with large datasets
        all_top_features = []
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:min(i+batch_size, n_samples)]
            # Get the full activations without topk restriction for final cluster assignment
            encoder_activations = sae.encoder(batch_X - sae.b_dec)
            # Get the index of the maximum activation for each example
            top_feature = encoder_activations.argmax(dim=1)
            all_top_features.append(top_feature.cpu())
            
            # Free memory
            del batch_X, encoder_activations, top_feature
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Combine all batches
        if len(all_top_features) > 1:
            cluster_labels = torch.cat(all_top_features, dim=0).numpy()
        else:
            cluster_labels = all_top_features[0].numpy()
    
    # Get cluster centers (decoder weights)
    # Each row of the decoder weight matrix corresponds to a cluster centroid
    cluster_centers = sae.W_dec.data.cpu().numpy()
    
    # Normalize cluster centers
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Save the SAE model after cluster_labels and cluster_centers are computed
    model_id = args.model.split('/')[-1].lower()
    os.makedirs('results/vars/saes', exist_ok=True)
    sae_save_path = f'results/vars/saes/sae_{model_id}_layer{args.layer}_clusters{n_clusters}.pt'

    # Embed the centering mean used for activation-cache construction so downstream can
    # reproduce the training activation space:
    #   x_norm = (x_raw - mean) / ||x_raw - mean||
    activation_mean_np = load_activation_mean(model_id=model_id, n_examples=int(args.n_examples), layer=int(args.layer))
    assert activation_mean_np.shape == (input_dim,), (
        f"activation_mean shape mismatch: {activation_mean_np.shape} vs expected {(input_dim,)}"
    )
    activation_mean_t = torch.from_numpy(activation_mean_np).to(torch.float32).cpu()

    torch.save({
        'encoder_weight': sae.encoder.weight.data.clone().cpu(),
        'encoder_bias': sae.encoder.bias.data.clone().cpu(),
        'decoder_weight': sae.W_dec.data.clone().cpu(),
        'b_dec': sae.b_dec.data.clone().cpu(),
        'input_dim': input_dim,
        'num_latents': n_clusters,
        'topk': topk,
        'loss': best_loss,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'activation_mean': activation_mean_t,
        'activation_mean_model_id': model_id,
        'activation_mean_layer': int(args.layer),
        'activation_mean_n_examples': int(args.n_examples),
    }, sae_save_path)
    print_and_flush(f"Saved SAE model to {sae_save_path}")
    
    # Clean up to free memory
    del sae, X
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print_and_flush(f"    Sparse autoencoder clustering completed in {time.time() - start_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers

# Dictionary mapping clustering method names to their implementations
CLUSTERING_METHODS = {
    'agglomerative': clustering_agglomerative,
    'pca_agglomerative': clustering_pca_agglomerative,
    'gmm': clustering_gmm,
    'pca_gmm': clustering_pca_gmm,
    'spherical_kmeans': clustering_spherical_kmeans,
    'pca_kmeans': clustering_pca_kmeans,
    'sae_topk': clustering_sae_topk
}