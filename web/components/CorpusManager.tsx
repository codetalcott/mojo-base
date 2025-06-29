// CorpusManager.tsx
// Web interface for managing incremental corpus updates
// Integrates with DaisyUI components for consistent styling

import React, { useState, useEffect } from 'react';

interface CorpusStats {
  total_vectors: number;
  total_files: number;
  total_chunks: number;
  projects: string[];
  languages: string[];
  corpus_size_mb: number;
}

interface ProjectFile {
  file_path: string;
  language: string;
  chunk_count: number;
  last_updated: string;
}

interface RecentUpdate {
  file_path: string;
  project: string;
  language: string;
  chunk_count: number;
  last_updated: string;
}

interface CorpusManagerProps {
  apiBaseUrl?: string;
  className?: string;
}

export const CorpusManager: React.FC<CorpusManagerProps> = ({
  apiBaseUrl = 'http://localhost:8001',
  className = ''
}) => {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [recentUpdates, setRecentUpdates] = useState<RecentUpdate[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProject, setUploadProject] = useState('');
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectPath, setNewProjectPath] = useState('');

  // Load initial data
  useEffect(() => {
    loadCorpusStats();
    loadRecentUpdates();
  }, []);

  const loadCorpusStats = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/corpus/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to load corpus stats:', error);
    }
  };

  const loadRecentUpdates = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/corpus/recent-updates?limit=10`);
      const data = await response.json();
      setRecentUpdates(data.recent_updates || []);
    } catch (error) {
      console.error('Failed to load recent updates:', error);
    }
  };

  const loadProjectFiles = async (projectName: string) => {
    try {
      const response = await fetch(`${apiBaseUrl}/corpus/files/${projectName}`);
      const data = await response.json();
      setProjectFiles(data.files || []);
    } catch (error) {
      console.error('Failed to load project files:', error);
    }
  };

  const handleProjectSelect = async (projectName: string) => {
    setSelectedProject(projectName);
    if (projectName) {
      await loadProjectFiles(projectName);
    } else {
      setProjectFiles([]);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile || !uploadProject) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('project', uploadProject);

    try {
      const response = await fetch(`${apiBaseUrl}/corpus/upload-file`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        await loadCorpusStats();
        await loadRecentUpdates();
        setUploadFile(null);
        setUploadProject('');
        
        // Show success message
        const modal = document.getElementById('upload_success_modal') as HTMLDialogElement;
        modal?.showModal();
      }
    } catch (error) {
      console.error('File upload failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddProject = async () => {
    if (!newProjectName || !newProjectPath) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/corpus/add-project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProjectName,
          path: newProjectPath
        })
      });

      if (response.ok) {
        await loadCorpusStats();
        await loadRecentUpdates();
        setNewProjectName('');
        setNewProjectPath('');
        
        // Show success message
        const modal = document.getElementById('project_success_modal') as HTMLDialogElement;
        modal?.showModal();
      }
    } catch (error) {
      console.error('Add project failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveProject = async (projectName: string) => {
    if (!confirm(`Are you sure you want to remove all files from project "${projectName}"?`)) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/corpus/remove-project/${projectName}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadCorpusStats();
        await loadRecentUpdates();
        if (selectedProject === projectName) {
          setSelectedProject('');
          setProjectFiles([]);
        }
      }
    } catch (error) {
      console.error('Remove project failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)} KB`;
    return `${mb.toFixed(1)} MB`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getLanguageBadgeClass = (language: string) => {
    const langColors: { [key: string]: string } = {
      typescript: 'badge-primary',
      javascript: 'badge-warning',
      python: 'badge-success',
      go: 'badge-info',
      rust: 'badge-error',
      java: 'badge-secondary'
    };
    return langColors[language] || 'badge-outline';
  };

  return (
    <div className={`max-w-6xl mx-auto p-6 ${className}`}>
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Corpus Manager</h1>
        <div className="badge badge-lg badge-primary">
          {stats ? `${stats.total_vectors} vectors` : 'Loading...'}
        </div>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="stat bg-base-100 rounded-lg shadow">
            <div className="stat-title">Total Vectors</div>
            <div className="stat-value text-primary">{stats.total_vectors.toLocaleString()}</div>
            <div className="stat-desc">{stats.total_chunks} chunks</div>
          </div>
          
          <div className="stat bg-base-100 rounded-lg shadow">
            <div className="stat-title">Projects</div>
            <div className="stat-value text-secondary">{stats.projects.length}</div>
            <div className="stat-desc">{stats.total_files} files</div>
          </div>
          
          <div className="stat bg-base-100 rounded-lg shadow">
            <div className="stat-title">Languages</div>
            <div className="stat-value text-accent">{stats.languages.length}</div>
            <div className="stat-desc">supported</div>
          </div>
          
          <div className="stat bg-base-100 rounded-lg shadow">
            <div className="stat-title">Corpus Size</div>
            <div className="stat-value text-info">{formatFileSize(stats.corpus_size_mb)}</div>
            <div className="stat-desc">in memory</div>
          </div>
        </div>
      )}

      {/* Management Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* File Upload */}
        <div className="card bg-base-100 shadow-lg">
          <div className="card-body">
            <h2 className="card-title">Upload File</h2>
            
            <div className="form-control">
              <label className="label">
                <span className="label-text">Project Name</span>
              </label>
              <input
                type="text"
                value={uploadProject}
                onChange={(e) => setUploadProject(e.target.value)}
                placeholder="Enter project name"
                className="input input-bordered"
              />
            </div>
            
            <div className="form-control">
              <label className="label">
                <span className="label-text">Code File</span>
              </label>
              <input
                type="file"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                accept=".js,.ts,.tsx,.jsx,.py,.go,.rs,.java,.cpp,.c,.h"
                className="file-input file-input-bordered"
              />
            </div>
            
            <div className="card-actions">
              <button
                onClick={handleFileUpload}
                disabled={!uploadFile || !uploadProject || isLoading}
                className="btn btn-primary"
              >
                {isLoading ? <span className="loading loading-spinner"></span> : 'Upload File'}
              </button>
            </div>
          </div>
        </div>

        {/* Add Project */}
        <div className="card bg-base-100 shadow-lg">
          <div className="card-body">
            <h2 className="card-title">Add Project</h2>
            
            <div className="form-control">
              <label className="label">
                <span className="label-text">Project Name</span>
              </label>
              <input
                type="text"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="my-awesome-project"
                className="input input-bordered"
              />
            </div>
            
            <div className="form-control">
              <label className="label">
                <span className="label-text">Project Path</span>
              </label>
              <input
                type="text"
                value={newProjectPath}
                onChange={(e) => setNewProjectPath(e.target.value)}
                placeholder="/path/to/project"
                className="input input-bordered"
              />
            </div>
            
            <div className="card-actions">
              <button
                onClick={handleAddProject}
                disabled={!newProjectName || !newProjectPath || isLoading}
                className="btn btn-secondary"
              >
                {isLoading ? <span className="loading loading-spinner"></span> : 'Add Project'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Project Management */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Project Browser */}
        <div className="card bg-base-100 shadow-lg">
          <div className="card-body">
            <h2 className="card-title">Browse Projects</h2>
            
            <div className="form-control">
              <select
                value={selectedProject}
                onChange={(e) => handleProjectSelect(e.target.value)}
                className="select select-bordered"
              >
                <option value="">Select a project...</option>
                {stats?.projects.map(project => (
                  <option key={project} value={project}>{project}</option>
                ))}
              </select>
            </div>
            
            {selectedProject && (
              <div className="mt-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold">{projectFiles.length} files</span>
                  <button
                    onClick={() => handleRemoveProject(selectedProject)}
                    className="btn btn-error btn-xs"
                  >
                    Remove Project
                  </button>
                </div>
                
                <div className="max-h-64 overflow-y-auto">
                  {projectFiles.map((file, index) => (
                    <div key={index} className="flex justify-between items-center py-2 border-b">
                      <div>
                        <div className="font-mono text-sm">{file.file_path}</div>
                        <div className="text-xs opacity-60">{file.chunk_count} chunks</div>
                      </div>
                      <div className={`badge ${getLanguageBadgeClass(file.language)}`}>
                        {file.language}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Recent Updates */}
        <div className="card bg-base-100 shadow-lg">
          <div className="card-body">
            <h2 className="card-title">Recent Updates</h2>
            
            <div className="max-h-80 overflow-y-auto">
              {recentUpdates.map((update, index) => (
                <div key={index} className="flex justify-between items-center py-3 border-b">
                  <div className="flex-1">
                    <div className="font-mono text-sm">{update.file_path}</div>
                    <div className="text-xs opacity-60">
                      {update.project} • {formatDate(update.last_updated)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`badge ${getLanguageBadgeClass(update.language)}`}>
                      {update.language}
                    </div>
                    <div className="badge badge-outline">
                      {update.chunk_count}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="card-actions">
              <button
                onClick={loadRecentUpdates}
                className="btn btn-ghost btn-sm"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Success Modals */}
      <dialog id="upload_success_modal" className="modal">
        <div className="modal-box">
          <h3 className="font-bold text-lg">Upload Successful!</h3>
          <p className="py-4">File has been added to the corpus and is now searchable.</p>
          <div className="modal-action">
            <form method="dialog">
              <button className="btn">Close</button>
            </form>
          </div>
        </div>
      </dialog>

      <dialog id="project_success_modal" className="modal">
        <div className="modal-box">
          <h3 className="font-bold text-lg">Project Added!</h3>
          <p className="py-4">Project has been scanned and added to the corpus.</p>
          <div className="modal-action">
            <form method="dialog">
              <button className="btn">Close</button>
            </form>
          </div>
        </div>
      </dialog>
    </div>
  );
};

export default CorpusManager;