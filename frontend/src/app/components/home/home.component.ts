import {Component} from '@angular/core';
import {Router} from "@angular/router";
import {AnalyzerService} from "../../services/analyzer.service";
import {ToastrService} from 'ngx-toastr';

interface Colors {
  [key: string]: string;
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  analyzing = false;
  analyzed = false;
  text = '';
  analyzedText = '';
  colors: Colors = {
    'B-MEDCOND': 'green'
  }

  constructor(
    private router: Router,
    private analyzerService: AnalyzerService,
    private notification: ToastrService,
  ) {
  }

  sanitizeHTML(htmlString: string): string {
    const allowedTagsRegex = /<(?!\/?(span|mark)\b)[^>]+>/gi;
    return htmlString.replace(allowedTagsRegex, '');
  }

  loadExample() {
    this.text = 'Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by sever' +
      'e lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chroni' +
      'c pain. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. Complicated by ' +
      'progressive lower extremity weakness and urinary retention. Patient initially presented with RLE weakness where h' +
      'is right knee gave out with difficulty walking and right anterior thigh numbness. MRI showed a spinal cord conus ' +
      'mass which was biopsied and found to be anaplastic astrocytoma. Therapy included field radiation t10-l1 followed ' +
      'by 11 cycles of temozolomide 7 days on and 7 days off. This was followed by CPT-11 Weekly x4 with Avastin Q2 week' +
      's/ 2 weeks rest and repeat cycle. On ROS, pt denies pain, lightheadedness, headache, neck pain, sore throat, recent ' +
      'illness or sick contacts, cough, shortness of breath, chest discomfort, heartburn, abd pain, n/v, diarrhea, ' +
      'constipation, dysuria. '
  }

  analyzeNote() {

    this.analyzing = true;

    if (this.text.toLowerCase() === 'github') {
      window.location.href = 'https://github.com/Padraig20/Applied-Deep-Learning-VU';
    } else if (this.text.toLowerCase() === 'linkedin') {
      window.location.href = 'https://linkedin.com/in/patrick-styll-009286244';
    } else {

      this.analyzerService.analyzeNote(this.text).subscribe({
        next: data => {
          console.log(data);
          this.analyzedText = '';
          const tokens: string[] = data.tokens;
          const entities: string[] = data.entities;

          let i = 0;
          while (i < tokens.length) {
            if (entities[i] === 'O') {
              this.analyzedText += tokens[i++];

              // Check if token is no dot, exclamatory mark, question mark, comma, or semicolon
              if (i < tokens.length && !['.', '!', '?', ',', ';'].includes(tokens[i])) {
                this.analyzedText += ' ';
              }
            } else {
              // Token is not 'O'; might be 'B' or 'I' entity but 'I' will be handled by the loop below
              if (entities[i].toString().startsWith('B-')) {
                this.analyzedText += '<mark class="highlight ' + this.colors[entities[i]] + '">' + tokens[i] + ' ';
                i++;
                while (i < tokens.length && entities[i].toString().startsWith('I-')) {
                  this.analyzedText += tokens[i] + ' ';
                  i++;
                }

                // Remove last space
                this.analyzedText = this.analyzedText.substring(0, this.analyzedText.length - 1);

                this.analyzedText += '<span class="descriptor">' + entities[i - 1].toString().substring(2) + '</span></mark> ';
              }
            }
          }

          this.notification.info('Successfully analyzed note!');
          console.log(this.analyzedText);
          this.analyzed = true;
          this.analyzing = false;
        },
        error: error => {
          console.log('Error analyzing note: ' + error);
          this.analyzing = false;
          this.notification.error('Error analyzing note');
        }
      });
    }
  }

  analyze() {
    this.analyzing = true;

    this.analyzerService.analyzeNote(this.text).subscribe({
      next: data => {
        this.analyzedText = '';
        console.log(data);
        const tokens = data.tokens[0];
        const entities = data.entities[0];
        let tmp = '';

        for (let i = 0; i < tokens.length; i++) {
          const token = tokens[i];
          const entity = entities[i];
          console.log(token);
          console.log(entity);

          if (entity === 'O') {
            this.analyzedText = this.analyzedText + (token + ' ');
          } else {

            if (entity === 'B-MEDCOND') {
              if ((i + 1) < tokens.length && entities[i + 1] === 'I-MEDCOND') {
                tmp = token;
              } else {
                this.analyzedText = this.analyzedText + '<span class="fw-bold bg-green-300 text-green-800 rounded px-1 py-1 m-1">' + (token) + '<span class="fw-light ml-2 text-green-950">MEDICAL CONDITION</span></span> ';
                tmp = '';
              }
            } else if (entity === 'I-MEDCOND') {
              tmp = tmp + (' ' + token);
              if ((i + 1) < tokens.length && entities[i + 1] !== 'I-MEDCOND' || i === tokens.length - 1) {
                this.analyzedText = this.analyzedText + '<span class="fw-bold bg-green-300 text-green-800 rounded px-1 py-1 m-1">' + (tmp) + '<span class="fw-light ml-2 text-green-950">MEDICAL CONDITION</span></span> ';
                tmp = '';
              }
            }
          }
        }
        this.notification.info('Successfully analyzed admission note!');
        console.log(this.analyzedText)
        this.analyzed = true;
        this.analyzing = false;
      },
      error: error => {
        console.log('Error analyzing note: ', error);
        this.analyzing = false;
        this.notification.error(error.error.message, 'Error analyzing note');
      }
    });
  }
}
