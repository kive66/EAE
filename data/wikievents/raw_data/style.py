        '''
        [
            {
                doc_id: document_id, 
                tokens: [ words, ...
                ],
                text: text,
                sentences , 
                entity_mentions: [
                    {
                        id,
                        sent_idx,
                        start,
                        end,
                        entity_type,
                        mention_type,
                        text
                    },
                    ...
                ],
                relation_mentions: [
                    ...
                ],
                event_mentions: [
                    {
                        id,
                        event_type,
                        trigger: {
                            start,
                            end,
                            text,
                            sent_idx
                        },
                        arguments: [
                            {
                                entity_id,
                                role,
                                text
                            },
                            ...
                        ]
                    }
                ]
            },
            ...
        ]    
        '''