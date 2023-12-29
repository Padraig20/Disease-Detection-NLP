import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {environment} from '../../environments/environment';

const baseUri = environment.backendUrl;

@Injectable({
  providedIn: 'root'
})
export class AnalyzerService {
  constructor(
    private http: HttpClient,
  ) { }

  /**
   * Analyze an admission note.
   *
   * @param note to analyze
   * @return observable list of diseases
   */
  analyzeNote(note: string): Observable<any> {
    return this.http.post<any>(baseUri + '/extract_entities', note);
  }
}
